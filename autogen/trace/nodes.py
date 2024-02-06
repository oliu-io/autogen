
import warnings
from typing import Optional, List, Dict, Callable, Union, Type, Any
import copy
from collections import defaultdict
import heapq

def node(message):
    """ Create a Node from a message. If message is already a Node, return it.
    This method is for the convenience of the user, it should be used over
    directly invoking Node."""
    if isinstance(message, Node):
        return message
    return Node(message)

NAME_SCOPES = []  # A stack of name scopes

class Graph:
    """ This a registry of all the nodes. All the nodes form a Directed Acyclic Graph.
    """
    TRACE = True  # When True, we trace the graph when creating MessageNode. When False, we don't trace the graph.

    def __init__(self):
        self._nodes = defaultdict(list)  # a lookup table to find nodes by name
        self._levels = defaultdict(list)  # a lookup table to find nodes at a certain level # TODO do we need this?

    def register(self, node):
        assert isinstance(node, Node)
        assert len(node.name.split(':'))==2
        name, id = node.name.split(':')
        if len(NAME_SCOPES)>0:
            name = NAME_SCOPES[-1] + '/' + name
        self._nodes[name].append(node)
        node._name = name + ':' + str(len(self._nodes[name])-1)
        self._levels[node._level].append(node)

    def get(self, name):
        name, id = name.split(':')
        return self._nodes[name][id]

    @property
    def leaves(self):
        return self._levels[0]

    def __str__(self):
        return str(self._nodes)

GRAPH = Graph()  # This is a global registry of all the nodes.

USED_NODES = list()  # A stack of sets. This is a global registry to track which nodes are read.

class AbstractNode:
    """ An abstract data node in a directed graph (parents <-- children).
    """
    def __init__(self, value, *, name=None, trainable=False) -> None:
        self._parents = []
        self._children = []
        self._level = 0  # leaves are at level 0
        self._name = str(type(value).__name__)+':0' if name is None else  name+':0'  # name:version
        if isinstance(value, Node):  # copy constructor
            self._data = copy.deepcopy(value._data)
            self._name = value._name
        else:
            self._data = value
        GRAPH.register(self)  # When created, register the node to the graph.

    @property
    def data(self):
        if len(USED_NODES)>0: # We're within trace_nodes context.
            USED_NODES[-1].add(self)
        return self._data

    @property
    def parents(self):
        return self._parents

    @property
    def children(self):
        return self._children

    @property
    def name(self):
        return self._name

    @property
    def level(self):
        return self._level

    @property
    def is_root(self):
        return len(self.parents) == 0

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child):
        assert child is not self, "Cannot add self as a child."
        assert isinstance(child, Node), f"{child} is not a Node."
        child.add_parent(self)

    def add_parent(self, parent):
        assert parent is not self, "Cannot add self as a parent."
        assert isinstance(parent, Node), f"{parent} is {type(parent)}, which is not a Node."
        parent._children.append(self)
        self._parents.append(parent)
        self._update_level(max(self._level, parent._level+1))  # Update the level, because the parent is added

    def _update_level(self, new_level):
        GRAPH._levels[self._level].remove(self)
        self._level = new_level
        GRAPH._levels[new_level].append(self)
        assert all([ len(GRAPH._levels[i])>0 for i in range(len(GRAPH._levels)) ]), "Some levels are empty."

    def __str__(self) -> str:
        # str(node) allows us to look up in the feedback dictionary easily
        return f'Node: ({self.name}, dtype={type(self._data)})'

    # def __eq__(self, other):
    #     # this makes it possible to store node as key in a dict
    #     # feedback[node.child] = feedback
    #     return hasattr(other, 'name') and self.name == other.name

    # def __hash__(self):
    #     # hash should do it over str(self) or repr(self)
    #     # choose whichever one makes most sense
    #     return hash(self.__str__())

    def __lt__(self, other):  # for heapq (since it is a min heap)
        return -self._level < -other._level

class Node(AbstractNode):
    """ Node for Autogen messages and prompts. Node behaves like a dict."""
    def __init__(self, value, *, name=None, trainable=False, description="This is a Node in a computational graph.") -> None:
        # TODO only take in a dict with a certain structure
        if isinstance(value, str):
            warnings.warn("Initializing a Node with str is deprecated. Use dict instead.")
        assert  isinstance(value, str) or isinstance(value, dict) or isinstance(value, Node), f"Value {value} must be a string, a dict, or a Node."
        super().__init__(value, name=name)
        self.trainable = trainable
        self._feedback = defaultdict(list)  # (analogous to gradient) this is the (synthetic) feedback from the user
        self._description = description # Infomation to describe of the node
        self._backwarded = False  # True if backward has been called

    @property
    def feedback(self):
        return self._feedback

    @property
    def description(self):
        return self._description  # TODO return a textual description of the node

    def _add_feedback(self, child, feedback):
        """ Add feedback from a child. """
        if self.feedback is None:
            raise AttributeError(f"{self} has been backwarded.")
        self.feedback[child].append(feedback)

    def _del_feedback(self):
        self._feedback = defaultdict(list)  # This saves memory and prevents backward from being called twice

    def backward(self, feedback: str, propagate, retain_graph=False):
        """ Backward pass.

            feedback: feedback given to the current node
            propagate: a function that takes in a node and a feedback, and returns a dict of {parent: parent_feedback}.

                def propagate(node, feedback):
                    return {parent: propagated feedback for parent in node.parents}

        """
        if self._backwarded:
            raise AttributeError(f"{self} has been backwarded.")

        assert type(feedback) == str, f"Feedback must be a string, but got {type(feedback)}."
        self._add_feedback('user', feedback)

        if len(self.parents) == 0:  # This is a leaf. Nothing to propagate
            return

        queue = [self]  # priority queue
        while True:
            try:
                node = heapq.heappop(queue)
                assert isinstance(node, Node)
                propagated_feedback = propagate(node)  # propagate information from child to parent
                for parent, parent_feedback in propagated_feedback.items():
                    parent._add_feedback(node, parent_feedback)
                    if len(parent.parents) > 0:
                        heapq.heappush(queue, parent)  # put parent in the priority queue

                node._del_feedback()  # delete feedback to save memory
                if not retain_graph and len(node.parents)>0:
                    node._backwarded = True  # set backwarded to True

            except IndexError:  # queue is empty
                break

    # We overload some magic methods to make it behave like a dict
    def __getattr__(self, name):
        if type(self.data) == dict:  # If attribute cannot be found, try to get it from the data
            return self.data.__getattribute__(name)
        else:
            raise AttributeError(f"{self} has no attribute {name}.")

    def __len__(self):
        return len(self.data)

    def __length_hint__(self):
        return NotImplemented

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        warnings.warn(f"Attemping to set {key} in {self.name}. In-place operation is not traced.")
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __reverse__(self):
        return reversed(self.data)

    def __contains__(self, key):
        return key in self.data

class ParameterNode(Node):
    # This is a shorthand of a trainable Node.
    def __init__(self, value, *, name=None, trainable=True, description="This is a ParameterNode in a computational graph.") -> None:
        super().__init__(value, name=name, trainable=trainable, description=description)

    def __str__(self) -> str:
        # str(node) allows us to look up in the feedback dictionary easily
        return f'ParameterNode: ({self.name}, dtype={type(self._data)})'


class MessageNode(Node):
    """ Output of an operator.

        description: a string to describe the operator it begins with
        [operator_name] and then describes the operator. When referring to
        inputs use the keys in args (if args is a dict), or the names of the
        nodes in args (if args is a list). Here're some examples:

        MessageNode(node_a, inputs=[node_a], description="[Identity] This is an idenity operator.")
        MessageNode(copy_node_a, inputs=[node_a], description="[Copy] This is a copy operator.")
        MesssageNode(1, inputs={'a':node_a, 'b':node_b}, description="[Add] This is an add operator of a and b.")
    """
    def __init__(self, value, *, inputs: Union[List[Node], Dict[str, Node]], description: str, name=None) -> None:
        super().__init__(value, name=name, description=description)
        # If not tracing, MessageNode would just behave like a Node.
        if GRAPH.TRACE:
            assert isinstance(inputs, list) or isinstance(inputs, dict)
            # If inputs is not a dict, we create a dict with the names of the nodes as keys
            if isinstance(inputs, list):
                _inputs = {}
                for i, v in enumerate(inputs):
                    _inputs[v.name] = v
                inputs = _inputs
            self._inputs = inputs
            # Add parents if we are tracing
            for k,v in self._inputs.items():
                assert isinstance(v, Node), f"Input {k} is not a Node."
                self.add_parent(v)

    def __str__(self) -> str:
        # str(node) allows us to look up in the feedback dictionary easily
        return f'MessageNode: ({self.name}, dtype={type(self._data)})'

    @property
    def data(self):  # MessageNode should act as immutable.
        return copy.deepcopy(super().data)


if __name__=='__main__':

    x = node('Node X')
    y = node('Node Y')
    z = MessageNode('Node Z', inputs={'x':x, 'y':y}, description='[Add] This is an add operator of x and y.')
    print(x.name, y.name)
    print([p.name for p in z.parents])