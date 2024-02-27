from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen.trace.nodes import MessageNode, USED_NODES, Node, supported_data_type, node
from dill.source import getsource
from collections.abc import Iterable


class trace_nodes:
    """ This is a context manager for keeping track which nodes are read/used in an operator."""
    def __enter__(self):
        nodes = set()
        USED_NODES.append(nodes)
        return nodes

    def __exit__(self, type, value, traceback):
        USED_NODES.pop()


def trace_operator(description, n_outputs=1):  # TODO add a dict to describe the inputs?
    def decorator(fun):
        """ This is a decorator to trace a function. The wrapped function returns a MessageNode.

            Args:
                fun (callable): the operator to be traced.
                description (str): a description of the operator; see the MessageNode for syntax.
                n_outputs (int); the number of outputs of the operator; default is 1.
        """
        assert callable(fun), "fun must be a callable."
        assert description is not None, "description must be provided."


        # TODO how to describe the mapping and inputs automatically?
        # description = getsource(fun)

        def wrapper(*args, **kwargs):
            """
            All nodes used in the operator fun are added to used_nodes during
            the execution. If the output is not a Node, we wrap it as a
            MessageNode, whose inputs are nodes in used_nodes.
            """
            with trace_nodes() as used_nodes:  # After exit, used_nodes contains the nodes whose data attribute is read in the operator fun.
                outputs = fun(*args, **kwargs)

            def wrap_output(output):
                if output is None:  # We keep None as None.
                    return output
                if len(used_nodes)==0:  # If no nodes are used, we don't need to wrap the output as a MessageNode.
                    return node(output)
                if isinstance(output, MessageNode):  # If the output is already a MessageNode, we don't need to wrap it.
                    return output
                # Else, we need to wrap the output as a MessageNode
                if supported_data_type(output):
                    return MessageNode(output, description=description, inputs=list(used_nodes))
                else:
                    raise NotImplementedError("The output of the operator is not supported.")
            if n_outputs==1:
                return wrap_output(outputs)
            else:
                return (wrap_output(outputs[i]) for i in range(n_outputs))

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