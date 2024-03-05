from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen.trace.nodes import Node, GRAPH
from autogen.trace.utils import for_all_methods
from autogen.agentchat.agent import Agent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.trace.trace_autogen import trace_ConversableAgent


## These are external APIs.

from autogen.trace.nodes import node
""" Create a Node from a message. If message is already a Node, return it.
    This method is for the convenience of the user, it should be used over
    directly invoking Node."""


def trace(cls, wrap_all_replies=True):
    """ A decorator to trace a ConversableAgent.

        Args:
            cls (subclass of ConversableAgent): The class to trace.
            wrap_all_replies (bool): If True, all registered reply_funcs would be wrappped by trace_op.
    """
    assert issubclass(cls, ConversableAgent), "cls must be a subclass of ConversableAgent."
    return trace_ConversableAgent(cls, wrap_all_replies=wrap_all_replies)
    # TODO: enable tracing other classes and functions

def trace_class(cls):
    assert issubclass(cls, ConversableAgent), "cls must be a subclass of ConversableAgent."
    return trace_ConversableAgent(cls, wrap_all_replies=False)

class stop_tracing():
    """ A contextmanager to disable tracing."""
    def __enter__(self):
        GRAPH.TRACE = False
    def __exit__(self, type, value, traceback):
        GRAPH.TRACE = True

def compatability(fun):
    """ This is a decorator to make a ConversableAgent compatabile with the trace framework. A method of the decorated class returns a MessageNode."""
    assert issubclass(fun, ConversableAgent), "fun must be a ConversableAgent or a callable method."
    traced_Cls = trace_ConversableAgent(fun)
    @for_all_methods
    def no_trace_decorator(fun):
        def wrapper(*args, **kwargs):
            with stop_tracing():
                return fun(*args, **kwargs)
        return wrapper
    return no_trace_decorator(traced_Cls)