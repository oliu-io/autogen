
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

    def __len__(self):
        # This is the number of nodes in the graph
        return sum([len(v) for v in self._nodes.values()])

GRAPH = Graph()  # This is a global registry of all the nodes.

USED_NODES = list()  # A stack of sets. This is a global registry to track which nodes are read.

from typing import TypeVar, Generic
T = TypeVar('T')
class AbstractNode(Generic[T]):
    """ An abstract data node in a directed graph (parents <-- children).
    """
    def __init__(self, value, *, name=None, trainable=False) -> None:
        self._parents = []
        self._children = []
        self._level = 0  # leaves are at level 0
        self._name = str(type(value).__name__)+':0' if name is None else  name+':0'  # name:version
        if isinstance(value, Node):  # just a reference
            self._data = value._data
            self._name = value._name
        else:
            self._data = value
        GRAPH.register(self)  # When created, register the node to the graph.

    @property
    def data(self):
        if len(USED_NODES)>0: # We're within trace_nodes context.
            USED_NODES[-1].add(self)
        return self.__getattribute__('_data')

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

    def clone(self):
        return copy.copy(self)

    def detach(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):  #
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_parents' or k == '_children':
                setattr(result, k, [])
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

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


# These are operators that do not change the data type and can be viewed as identity operators.
IDENTITY_OPERATORS = ('identity', 'Copy', 'message_to_dict', 'oai_message')

import re
def get_operator_type(description):
    """ Extra the operator type from the description. """
    match = re.search(r"\[([^[\]]+)\]", description)  # TODO. right pattern?
    if match:
        operator_type = match.group(1)
        # TODO check admissible types
        return operator_type
    else:
        raise ValueError(f"The description '{description}' must contain the operator type in square brackets.")

def supported_data_type(value):
    return isinstance(value, bool) or isinstance(value, str) or isinstance(value, dict) or isinstance(value, Node)

class Node(AbstractNode[T]):
    """ Node for Autogen messages and prompts. Node behaves like a dict."""
    def __init__(self, value, *, name=None, trainable=False, description="[Node] This is a node in a computational graph.") -> None:
        # TODO only take in a dict with a certain structure
        if isinstance(value, str):
            warnings.warn("Initializing a Node with str is deprecated. Use dict instead.")
        # assert supported_data_type(value), f"Value {value} must be a bool, a string, a dict, or a Node."
        super().__init__(value, name=name)
        self.trainable = trainable
        self._feedback = defaultdict(list)  # (analogous to gradient) this is the feedback from the user. Each key is a child and the value is a list of feedbacks from the child.
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

    def backward(self, feedback: str, propagate, retain_graph=False, visualize=False, reverse_plot=False, print_limit=100):
        """ Backward pass.

            feedback: feedback given to the current node
            propagate: a function that takes in a node and a feedback, and returns a dict of {parent: parent_feedback}.

                def propagate(node, feedback):
                    return {parent: propagated feedback for parent in node.parents}

            visualize: if True, plot the graph using graphviz
            reverse_plot: if True, plot the graph in reverse order (from child to parent).
            print_limit: the maximum number of characters to print in the graph.

        """

        assert type(feedback) == str, f"Feedback must be a string, but got {type(feedback)}."

        # Setup for visualization
        digraph = None
        if visualize:
            from graphviz import Digraph
            digraph = Digraph()
            get_name = lambda x: x.name.replace(":", "")  # using colon in the name causes problems in graphviz
            def get_label(x):
                text = get_name(x)+'\n'+x.description+'\n'
                content = str(x.data['content'] if isinstance(x.data, dict) else x.data)
                if len(content) > print_limit:
                    content = content[:print_limit] + '...'
                return text + content
            visited = set()

        # Check for root node with no parents
        if self._backwarded:
            raise AttributeError(f"{self} has been backwarded.")
        self._add_feedback('user', feedback)
        if len(self.parents) == 0:  # This is a root. Nothing to propagate
            if visualize:
                digraph.node(get_name(node), label=get_label(node))
            self._backwarded = not retain_graph
            return

        # TODO optimize for efficiency
        # TODO check memory leak
        queue = [self]  # priority queue
        while True:
            try:
                node = heapq.heappop(queue)
                # Each node is a MessageNode, which has at least one parent.
                assert len(node.parents) > 0 and isinstance(node, MessageNode)
                if node._backwarded:
                    raise AttributeError(f"{node} has been backwarded.")

                # Propagate information from child to parent
                propagated_feedback = propagate(node)
                for parent, parent_feedback in propagated_feedback.items():
                    parent._add_feedback(node, parent_feedback)
                    # Put parent in the queue if it has not been visited and it's not a root
                    if len(parent.parents) > 0 and parent not in queue: # and parent not in queue:
                        heapq.heappush(queue, parent)  # put parent in the priority queue

                    if visualize:
                        # Plot the edge from parent to node
                        # Bypass chain of identity operators (for better visualization)
                        while get_operator_type(parent.description) in IDENTITY_OPERATORS:
                            assert len(parent.parents)==1  # identity operators should have only one parent
                            visited.add(get_name(parent)) # skip this node in visualization
                            parent = parent.parents[0]

                        edge = (get_name(node), get_name(parent)) if reverse_plot else (get_name(parent), get_name(node))
                        # Just plot the edge once, since the same node can be
                        # visited multiple times (e.g., when that node has
                        # multiple children).
                        if edge not in visited and get_name(node) not in visited:
                            digraph.edge(*edge)
                            visited.add(edge)
                            digraph.node(get_name(node), label=get_label(node))
                            digraph.node(get_name(parent), label=get_label(parent))

                node._backwarded = not retain_graph  # set backwarded to True

            except IndexError:  # queue is empty
                break

        return digraph

    # TODO remove these
    # We overload some magic methods to make it behave like a dict
    def __getattr__(self, name):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        data = self.__getattribute__('data')
        try:  # If attribute cannot be found, try to get it from the data
            return data.__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"{self} has no attribute {name}.")

    def __bool__(self):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        return bool(self.data)

    def __len__(self):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        return len(self.data)

    def __length_hint__(self):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        return NotImplemented

    def __getitem__(self, key):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        return self.data[key]

    def __setitem__(self, key, value):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        self._data[key] = value

    def __delitem__(self, key):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        del self.data[key]

    def __iter__(self):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        return iter(self.data)

    def __reverse__(self):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        return reversed(self.data)

    def __contains__(self, key):
        warnings.warn(f"Deprecated: Attemping to access data in {self}.")
        return key in self.data

class ParameterNode(Node[T]):
    # This is a shorthand of a trainable Node.
    def __init__(self, value, *, name=None, trainable=True, description="This is a ParameterNode in a computational graph.") -> None:
        super().__init__(value, name=name, trainable=trainable, description=description)

    def __str__(self) -> str:
        # str(node) allows us to look up in the feedback dictionary easily
        return f'ParameterNode: ({self.name}, dtype={type(self._data)})'

class MessageNode(Node[T]):
    """ Output of an operator.

        description: a string to describe the operator it begins with
        [operator_name] and then describes the operator. When referring to
        inputs use the keys in args (if args is a dict), or the names of the
        nodes in args (if args is a list). Here're some examples:

        MessageNode(node_a, inputs=[node_a], description="[identity] This is an idenity operator.")
        MessageNode(copy_node_a, inputs=[node_a], description="[copy] This is a copy operator.")
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

    def _add_feedback(self, child, feedback):
        """ Add feedback from a child. """
        if self.feedback is None:
            raise AttributeError(f"{self} has been backwarded.")
        self.feedback[child] = [feedback] # Only one feedback is allowed for MessageNode

    # @property
    # def data(self):  # MessageNode should act as immutable.
    #     return super().data #copy.deepcopy(super().data)


if __name__=='__main__':

    x = node('Node X')
    y = node('Node Y')
    z = MessageNode('Node Z', inputs={'x':x, 'y':y}, description='[Add] This is an add operator of x and y.')
    print(x.name, y.name)
    print([p.name for p in z.parents])

    x : AbstractNode[str] = node('Node X')
    x : Node[str] = node('Node X')
    x: ParameterNode[str] = ParameterNode('Node X', trainable=True)
    x: MessageNode[str] = MessageNode('Node X', inputs={'x':x, 'y':y}, description='[Add] This is an add operator of x and y.')