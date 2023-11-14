
import warnings
from typing import Optional, List, Dict, Callable, Union, Type, Any
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.agent import Agent
import copy
# from autogen.trace.nodes import Node
from collections import defaultdict, deque



class Graph:
    """ Directed Acyclic Graph. A global registry of all the nodes.
    """
    TRACE = True

    def __init__(self):
        self._nodes = defaultdict(list)  # a lookup table to find nodes by name
        self._levels = defaultdict(list)  # a lookup table to find nodes at a certain level # TODO do we need this?

    def register(self, node):
        assert isinstance(node, Node)
        assert len(node.name.split(':'))==2
        name, id = node.name.split(':')
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

GRAPH = Graph()

class AbstractNode:
    """ An abstract data node in a directed graph (parents <-- children).
    """
    def __init__(self, value, *, name=None, trainable=False) -> None:
        self._parents = []
        self._children = []
        self._level = 0  # leaves are at level 0
        self._name = str(type(value).__name__)+':0' if name is None else  name+':0'
        if isinstance(value, Node):  # copy constructor
            self._data = copy.deepcopy(value._data)
            self._name = value._name
        else:
            self._data = value
        GRAPH.register(self)

    @property
    def data(self):
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

    def add_child(self, child):
        assert child is not self, "Cannot add self as a child."
        assert isinstance(child, Node), f"{child} is not a Node."
        child.add_parent(self)

    def add_parent(self, parent):
        assert parent is not self, "Cannot add self as a parent."
        assert isinstance(parent, Node), f"{parent} is not a Node."
        parent._children.append(self)
        self._parents.append(parent)
        self._update_level(max(self._level, parent._level+1))  # Update the level, because the parent is added

    def _update_level(self, new_level):
        GRAPH._levels[self._level].remove(self)
        self._level = new_level
        GRAPH._levels[new_level].append(self)
        assert all([ len(GRAPH._levels[i])>0 for i in range(len(GRAPH._levels)) ]), "Some levels are empty."

    def __str__(self) -> str:
        return f'Node: ({self.name}, dtype={type(self.data)})'


class Node(AbstractNode):
    """ Node for Autogen messages and prompts"""
    def __init__(self, value, *, name=None, trainable=False) -> None:
        # TODO only take in a dict with a certain structure
        if isinstance(value, str):
            warnings.warn("Initializing a Node with str is deprecated. Use dict instead.")
        assert  isinstance(value, str) or isinstance(value, dict) or isinstance(value, Node), f"Value {value} must be a string, a dict, or a Node."
        super().__init__(value, name=name)
        self.trainable = trainable
        self._feedback = dict()  # (analogous to gradient) this is the (synthetic) feedback from the user

    @property
    def _has_all_feedback(self):
        """ Whether the node has feedback from all children. """
        for child in self.children:
            if self._feedback.get(child.name) is None:
                return False
        return True

    def _add_feedback(self, child, feedback):
        """ Add feedback from a child. """
        self._feedback[child.name] = feedback

    def _del_feedback(self):
        self._feedback = None  # This saves memory and prevents backward from being called twice

    # We overload some magic methods to make it behave like a dict
    def __getattr__(self, name):
        if type(self._data) == dict:  # If attribute cannot be found, try to get it from the data
            return self._data.__getattribute__(name)
        else:
            raise AttributeError(f"{self} has no attribute {name}.")

    def __len__(self):
        return len(self._data)

    def __length_hint__(self):
        return NotImplemented

    def __getitem__(self, key):
        warnings.warn(f"Attempting to get {key} from {self.name}.")
        return self._data[key]

    def __setitem__(self, key, value):
        warnings.warn(f"Attemping to set {key} in {self.name}.")
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __reverse__(self):
        return reversed(self._data)

    def __contains__(self, key):
        return key in self._data

# TODO
class ParameterNode(Node):
    # This is a shorthand of a trainable Node.
    def __init__(self, value, *, name=None, trainable=True) -> None:
        super().__init__(value, name=name, trainable=trainable)


class MessageNode(Node):
    """ Output of an operator. """
    def __init__(self, value, mapping, *, args=None, kwargs=None, name=None) -> None:
        super().__init__(value, name=name)
        if GRAPH.TRACE:
            self._mapping = mapping
            self._args = () if args is None else args
            self._kwargs = {} if kwargs is None else kwargs
            for v in self._args:
                self.add_parent(v)
            for v in self._kwargs.values():
                self.add_parent(v)

    def backward(self, feedback, propagate, retain_graph=False):
        """ Backward pass. """
        if self._feedback is None:  # This node has been backwarded
            raise AttributeError(f"{self} has been backwarded.")

        self._feedback['user'] = feedback
        queue = deque([self])
        while True:
            try:
                node = queue.pop()  # node has accumulated feedback from all children
                assert node._has_all_feedback, f"{node} does not have feedback from all children."
                for parent in node.parents:
                    parent_feedback = propagate(node, parent, feedback)  # propagate information from child to parent
                    parent._add_feedback(node, parent_feedback)
                    if parent._has_all_feedback:
                        queue.append(parent)
                if not retain_graph and len(node.parents)>0:
                    node._del_feedback()
            except IndexError:  # queue is empty
                break