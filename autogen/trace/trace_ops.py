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

def trace_op(description, n_outputs=1, node_dict=None):
    def decorator(fun):
        """ This is a decorator to trace a function. The wrapped function returns a MessageNode.

            Args:
                fun (callable): the operator to be traced.
                description (str): a description of the operator; see the MessageNode for syntax.
                n_outputs (int); the number of outputs of the operator; default is 1.
                node_dict (dict|str):
                    'auto' : the inputs are represented as a list of nodes.
                    'auto': the inputs are represented as a dictionary, where the keys are the parameter names and the values are the nodes.
                    dict : a dictionary to describe the inputs, where the key is a node used in this operator and the value is the node's name as described in description ; when node_dict is provided, all the used_nodes need to be in node_dict. Providing node_dict can give a correspondance between the inputs and the description of the operator.

        """
        assert callable(fun), "fun must be a callable."
        assert description is not None, "description must be provided."
        assert isinstance(node_dict, dict) or (node_dict is None) or (node_dict=='auto'),  "node_dict must be a dictionary or None or 'auto."

        # TODO how to describe the mapping and inputs automatically?
        # description = getsource(fun)

        operator_name = get_operator_name(description)  # TODO using a dict?
        def wrap_output(output, inputs: Union[List[Node],Dict[str,Node]]):
            """ Wrap the output as a MessageNode of inputs as the parents."""
            if output is None:  # We keep None as None.
                return output
            # Some nodes are used in the operator fun, we need to wrap the output as a MessageNode.
            if isinstance(output, Node):  # If the output is already a Node, we don't need to wrap it.
                return output  # NOTE User who implements fun is responsible for the graph structure.
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
            if node_dict is None:
                # If no information is provided, we represent inputs as list.
                # MessageNode will convert inputs as dict based on the names of
                # the nodes in used_nodes
                inputs = list(used_nodes)
            else:  # Otherwise we represent inputs as dict
                assert node_dict == 'auto' or isinstance(node_dict, dict)
                spec = inspect.getcallargs(fun, *args, **kwargs) # Read it from the input signature
                if isinstance(node_dict, dict):
                    spec.update(node_dict)
                assert isinstance(spec, dict)
                # Makre sure all nodes in used_nodes are in spec
                assert all([node in spec.values() for node in used_nodes]), "All used_nodes must be in the spec."
                inputs = {k:v for k,v in spec.items() if v in used_nodes}

            if n_outputs==1:
                nodes =  wrap_output(outputs, inputs)
                parents = set(nodes.parents) if isinstance(nodes, Node) else set()
            else:
                nodes = tuple(wrap_output(outputs[i], inputs) for i in range(n_outputs))
                parents = set.union(*[set(node.parents) if isinstance(nodes, Node) else set() for node in nodes])

            # Make sure all nodes in used_nodes are in the parents of the returned node.
            if not all([node in parents for node in used_nodes]):
                raise ValueError(f"Not all nodes used in the operator {fun} are specified as inputs of the returned node.")
            return nodes

        return wrapper
    return decorator


class NodeContainer:
    pass

def apply_op(op, output, *args, **kwargs):
    """ Apply an op to container of Nodes.

        Args:
            op (callable): the operator to be applied.
            output (Any): the container to be updated.
            *args (Any): the positional inputs of the operator.
            **kwargs (Any): the keyword inputs of the operator.
    """

    inputs = list(args) + list(kwargs.values())
    containers = [x for x in inputs if not isinstance(x, Node)]
    if len(containers)==0:  # all inputs are Nodes, we just apply op
        return op(*args, **kwargs)
    # # there is at least one container
    # output = copy.deepcopy(containers[0])  # this would be used as the template of the output

    def admissible_type(x,base):
        return type(x)==type(base) or isinstance(x,Node)
    assert all(admissible_type(x, output) for x in inputs)  # All inputs are either Nodes or the same type as output

    if isinstance(output,list) or isinstance(output, tuple):
        output = list(output)
        assert all(isinstance(x,Node) or len(output)==len(x) for x in inputs), f"output {output} and inputs {inputs} are of different lengths."
        for k in range(len(output)):
            _args = [ x if isinstance(x, Node) else x[k] for x in args]
            _kwargs = {kk: vv if isinstance(vv, Node) else vv[k] for kk, vv in kwargs.items()}
            output[k] = apply_op(op, output[k], *_args, **_kwargs)

    elif isinstance(output, dict):
        for k,v in output.items():
            _args = [ x if isinstance(x, Node) else x[k] for x in args]
            _kwargs = {kk: vv if isinstance(vv, Node) else vv[k] for kk, vv in kwargs.items()}
            output[k] = apply_op(op, output[k], *_args, **_kwargs)

    elif isinstance(output, NodeContainer): # this is a NodeContainer object instance
        for k,v in output.__dict__.items():
            _args = [ x if isinstance(x, Node) else getattr(x,k) for x in args]
            _kwargs = {kk: vv if isinstance(v, Node) else getattr(vv,k) for kk, vv in kwargs.items()}
            new_v = apply_op(op,  v, *_args, **_kwargs)
            setattr(output, k ,new_v )
    else:
        pass
    return output


if __name__=='__main__':
    from autogen.trace.trace import node
    x = node('hello')
    @trace_op('[Custom] This is a test function.')
    def test(x):
        return x.data+' world'

    y = test(x)
    print(y)
    print('Parents', y.parents)
    print('Children', y.children)
    print('Level', y._level)