from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen.trace.nodes import MessageNode, USED_NODES
from dill.source import getsource
import re


class trace_nodes:
    """ This is a context manager for keeping track which nodes are read/used in an operator."""
    def __enter__(self):
        nodes = set()
        USED_NODES.append(nodes)
        return nodes

    def __exit__(self, type, value, traceback):
        USED_NODES.pop()


def trace_operator(description):  # TODO add a dict to describe the inputs?
    def decorator(fun):
        """ This is a decorator to trace a function. The wrapped function returns a MessageNode.

            Args:
                fun (callable): the operator to be traced.
                description (str): a description of the operator; see the MessageNode for syntax.
        """
        assert callable(fun), "fun must be a callable."
        assert description is not None, "description must be provided."

        match = re.search(r"\[([^[\]]+)\]", description)  # TODO. right pattern?
        if match:
            operator_type = match.group(1)
            # TODO check admissible types
        else:
            raise ValueError(f"The description '{description}' must contain the operator type in square brackets.")


        # TODO how to describe the mapping and inputs automatically?
        # description = getsource(fun)

        def wrapper(*args, **kwargs):
            """
            All nodes used in the operator fun are added to used_nodes during
            the execution. If the output is not a Node, we wrap it as a
            MessageNode, whose inputs are nodes in used_nodes.
            """
            with trace_nodes() as used_nodes:  # After exit, used_nodes contains the nodes whose data attribute is read in the operator fun.
                output = fun(*args, **kwargs)
            if output is not None and not isinstance(output, MessageNode):
                output = MessageNode(output, description=description, inputs=list(used_nodes))
            return output
        return wrapper
    return decorator


if __name__=='__main__':
    from autogen.trace.trace import node
    x = node('hello')
    @trace_operator('[Custom] This is a test function.')
    def test(x):
        return x.data+' world'

    y = test(x)
    print(y)
    print('Parents', y.parents)
    print('Children', y.children)
    print('Level', y._level)