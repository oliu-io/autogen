import copy

from dataclasses import dataclass
import warnings


from typing import Optional, List, Dict, Callable, Union, Type, Any
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.agent import Agent
import copy
# from autogen.trace.nodes import Node


class Registry:
    """ A global registry of all the nodes. """

    def __init__(self):
        self._nodes = {}

    def register(self, node):
        assert isinstance(node, Node)
        assert len(node.name.split(':'))==2
        if node.name in self._nodes:
            # increment the id
            name, id = node.name.split(':')
            node._name = name + ':' + str(int(id)+1)
        self._nodes[node.name] = node

    def get(self, name):
        return self._nodes[name]

    def __str__(self):
        return str(self._nodes)

GRAPH = Registry()

class Node:
    """ An abstract data node in a polytree. """
    def __init__(self, value, *, name=None) -> None:
        assert value is not None, "value cannot be None."
        self._parent = None
        self._children = []
        self._level = 0
        self._name = str(type(value).__name__)+':0' if name is None else  name+':0'
        if isinstance(value, Node):  # copy constructor
            self._data = copy.deepcopy(value._data)
            self._name = value._name
        else:
            # TODO add assertion on value type
            self._data = value

        GRAPH.register(self)

    @property
    def data(self):
        return self._data

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @property
    def name(self):
        return self._name

    @parent.setter
    def parent(self, parent):
        assert isinstance(parent, Node), f"{parent} is not a Node."
        assert self._parent is None, f"{self.name} already has a parent."
        parent.add_child(self)


    def add_child(self, child):
        assert isinstance(child, Node), f"{child} is not a Node."
        assert child not in self.children, f"{child} is already a child of {self}."
        self._children.append(child)
        self._level = max(self._level, child._level+1)
        child._parent = self

    def __str__(self) -> str:
        return f'Node: ({self.name}, dtype={type(self.data)})'

    def __getattr__(self, name):
        # If attribute cannot be found, try to get it from the data
        warnings.warn(f"Attribute {name} not found in {self.name}. Attempting to get it from the data.")
        return self._data.__getattribute__(name)

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        warnings.warn(f"Attempting to get {key} from {self.name}.")
        return self._data[key]

    def __setitem__(self, key, value):
        warnings.warn(f"Attemping to set {key} in {self.name}.")
        self._data[key] = value



# TODO
class ParameterNode(Node):
    # These are the trainable nodes
    pass


class MessageNode(Node):
    """ Output of an operator. """
    def __init__(self, value, mapping, *, args=None, kwargs=None, name=None) -> None:
        super().__init__(value, name=name)
        self._mapping = mapping
        self._args = () if args is None else args
        self._kwargs = {} if kwargs is None else kwargs
        for v in self._args:
            self.add_child(v)
        for v in self._kwargs.values():
            self.add_child(v)

    # def __getattr__(self, name):
    #     # If attribute cannot be found, try to get it from the data
    #     attr = self._data.__getattribute__(name)  # TODO
    #     # TODO add assertion
    #     if callable(attr):
    #         return trace(attr) # TODO
    #     else:
    #         output = Node(attr)
    #         output.register_mapping(f"{output.name}={self.name}.{name}", self)
    #         return attr
