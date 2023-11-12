import copy

from dataclasses import dataclass


from typing import Optional, List, Dict, Callable, Union, Type, Any
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.agent import Agent
import copy
# from autogen.trace.nodes import Node


import inspect

def trace(fun):
    # trace a function
    # The wrapped function returns a message node
    def wrapper(*args, **kwargs):
        # call the function with the data
        result = fun(*args, **kwargs)
        # wrap the inputs and outputs as Nodes if they're not
        m_args = (Node(v) for v in args if not isinstance(v, Node))
        m_kwargs = {k: Node(v) for k, v in kwargs.items() if not isinstance(v, Node)}
        mapping = inspect.getsource(fun)  # TODO how to describe the mapping and inputs?
        # get the source code
        # inspect.getdoc(fun)
        m_result = MessageNode(result, mapping, m_args, m_kwargs) if not isinstance(result, MessageNode) else result # TODO
        return m_result
    return wrapper


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


    # # TODO
    # def __add__(self, other):
    #     if not isinstance(other, Node):
    #         other = Node(other)
    #     mapping = f"output=x+y"
    #     breakpoint()
    #     output = MessageNode(self.data+other.data, mapping, kwargs=dict(x=self, y=other))
    #     return output



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


if __name__=='__main__':


    x = Node('hello')

    @trace
    def test(x):
        return x+' world'

    y = test(x)
    print(y)
    print('Parent', y.parent)
    print('Children', y.children)
    print('Level', y._level)
