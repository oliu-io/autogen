from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen.trace.nodes import MessageNode, USED_NODES, Node, node, get_operator_name
from dill.source import getsource
from collections.abc import Iterable
import inspect

import warnings
class trace_nodes:
    """ This is a context manager for keeping track which nodes are read/used in an operator."""
    def __enter__(self):
        nodes = set()
        USED_NODES.append(nodes)
        return nodes

    def __exit__(self, type, value, traceback):
        USED_NODES.pop()

# TODO rename to trace_op
def trace_operator(description, n_outputs=1, node_dict='auto'):  # TODO add a dict to describe the inputs?
    def decorator(fun):
        """ This is a decorator to trace a function. The wrapped function returns a MessageNode.

            Args:
                fun (callable): the operator to be traced.
                description (str): a description of the operator; see the MessageNode for syntax.
                n_outputs (int); the number of outputs of the operator; default is 1.
                node_dict (dict|str):
                    'auto' : the inputs are represented as a list of nodes.
                    'signature': the inputs are represented as a dictionary, where the keys are the parameter names and the values are the nodes.
                    dict : a dictionary to describe the inputs, where the key is a node used in this operator and the value is the node's name as described in description ; when node_dict is provided, all the used_nodes need to be in node_dict. Providing node_dict can give a correspondance between the inputs and the description of the operator.

        """
        assert callable(fun), "fun must be a callable."
        assert description is not None, "description must be provided."
        assert type(node_dict) is dict or node_dict in ('signature', 'auto'), "node_dict must be a dictionary or None or 'auto."

        # TODO how to describe the mapping and inputs automatically?
        # description = getsource(fun)

        operator_name = get_operator_name(description)  # TODO using a dict?
        def wrap_output(output, inputs: Union[List[Node],Dict[str,Node]]):
            """ Wrap the output as a MessageNode of inputs as the parents."""
            if output is None:  # We keep None as None.
                return output
            if len(inputs)==0:  # If no nodes are used, we don't need to wrap the output as a MessageNode.
                return Node(output, name=operator_name)
            # Some nodes are used in the operator fun, we need to wrap the output as a MessageNode.
            if isinstance(output, MessageNode):  # If the output is already a Node, we don't need to wrap it.
                if not all([node in output.parents for node in inputs]):
                     warnings.warn(f"Not all nodes used in the operator {fun} are part of the inputs of the output. The output may not be consistent with the inputs.")
                return output  # NOTE User who implements fun is responsible for the graph structure.
            # Else, we need to wrap the output as a MessageNode
            return MessageNode(output, description=description, inputs=inputs, name=operator_name)


        def wrapper(*args, **kwargs):
            """
            All nodes used in the operator fun are added to used_nodes during
            the execution. If the output is not a Node, we wrap it as a
            MessageNode, whose inputs are nodes in used_nodes.
            """
            # After exit, used_nodes contains the nodes whose data attribute is read in the operator fun.
            with trace_nodes() as used_nodes:
                outputs = fun(*args, **kwargs)

            # Construct the inputs of the MessageNode from the set used_nodes
            if node_dict=='auto':
                # If no information is provided, we represent inputs as list.
                # MessageNode will convert inputs as dict based on the names of
                # the nodes in used_nodes
                inputs = list(used_nodes)
            else:  # Otherwise we represent inputs as dict
                if node_dict=='signature':  # Read it from the input signature
                    spec = inspect.getcallargs(fun, *args, **kwargs)
                else:
                    spec = node_dict  # TODO Is this easy to specify?
                assert isinstance(spec, dict)
                # Makre sure all nodes in used_nodes are in spec
                assert all([node in spec.values() for node in used_nodes]), "All used_nodes must be in the spec."
                inputs = {k:v for k,v in spec.items() if v in used_nodes}

            if n_outputs==1:
                return wrap_output(outputs, inputs)
            else:
                return (wrap_output(outputs[i], inputs) for i in range(n_outputs))

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