from curses import wrapper
from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen import trace
from autogen.trace.modules import apply_op, to_data, Module
from autogen.trace.nodes import MessageNode, USED_NODES, Node, ParameterNode, node, get_operator_name
from dill.source import getsource
from collections.abc import Iterable
import inspect
import functools
import re


class trace_nodes:
    """This is a context manager for keeping track which nodes are read/used in an operator."""

    def __enter__(self):
        nodes = set()
        USED_NODES.append(nodes)
        return nodes

    def __exit__(self, type, value, traceback):
        USED_NODES.pop()


def trace_op(description=None, n_outputs=1, node_dict=None, wrap_output=True, unpack_input=True, variable=False):
    """
    Wrap a function as a FunModule, which returns node objects.
    The input signature to the wrapped function stays the same.
    """

    def decorator(fun):
        return FunModule(
            fun=fun,
            description=description,
            n_outputs=n_outputs,
            node_dict=node_dict,
            wrap_output=wrap_output,
            unpack_input=unpack_input,
            variable=variable,
        )

    return decorator


class FunModule(Module):
    """This is a decorator to trace a function. The wrapped function returns a MessageNode.

    Args:
        fun (callable): the operator to be traced.
        description (str): a description of the operator; see the MessageNode for syntax.
        n_outputs (int); the number of outputs of the operator; default is 1.
        node_dict (dict|str):
            None : the inputs are represented as a list of nodes.
            'auto': the inputs are represented as a dictionary, where the keys are the parameter names and the values are the nodes.
            dict : a dictionary to describe the inputs, where the key is a node used in this operator and the value is the node's name as described in description ; when node_dict is provided, all the used_nodes need to be in node_dict. Providing node_dict can give a correspondence between the inputs and the description of the operator.
        wrap_output (bool): if True, the output of the operator is wrapped as a MessageNode; if False, the output is returned as is if the output is a Node.
        unpack_input (bool): if True, the input is extracted from the container of nodes; if False, the inputs are passed directly to the underlying function.
        variable (bool): if True, the block of code is treated as a variable in the optimization

    """

    def __init__(
        self,
        fun: Callable,
        description: str = None,
        n_outputs: int = 1,
        node_dict: Union[dict, None, str] = "auto",
        wrap_output: bool = True,
        unpack_input: bool = True,
        variable=False,
    ):
        assert callable(fun), "fun must be a callable."
        assert (
            isinstance(node_dict, dict) or (node_dict is None) or (node_dict == "auto")
        ), "node_dict must be a dictionary or None or 'auto."
        match = re.search(r"\s*@trace_op\(.*\)\n\s*(def.*)", inspect.getsource(fun), re.DOTALL)
        source = match.group(1).strip()
        self.info = dict(
            fun_name=fun.__qualname__,
            doc=fun.__doc__,
            signature=inspect.signature(fun),
            source=source,
        )
        if description is None:
            # Generate the description from the function name and docstring.
            description = f"[{self.info['fun_name']}] {self.info['doc']}."
        self._fun = fun
        self.node_dict = node_dict
        self.info["node_dict"] = node_dict
        self.description = description
        self.n_outputs = n_outputs
        self.wrap_output = wrap_output
        self.unpack_input = unpack_input
        self.parameter = None
        if variable:
            self.parameter = ParameterNode(self.info["source"], name="__code")

    @property
    def fun(self, *args, **kwargs):
        # This is called within trace_nodes context manager.
        if self.parameter is None:
            return self._fun
        else:
            code = self.parameter._data  # This is not traced, but we will add this as the parent later.
            exec(code)  # define the function
            return locals()[self.info["fun_name"]]

    @property
    def name(self):
        return get_operator_name(self.description)

    def forward(self, *args, **kwargs):
        """
        All nodes used in the operator fun are added to used_nodes during
        the execution. If the output is not a Node, we wrap it as a
        MessageNode, whose inputs are nodes in used_nodes.
        """
        # After exit, used_nodes contains the nodes whose data attribute is read in the operator fun.
        with trace_nodes() as used_nodes:
            _args, _kwargs = args, kwargs
            if self.unpack_input:  # extract data from container of nodes
                _args = to_data(args)
                _kwargs = to_data(kwargs)
            outputs = self.fun(*_args, **_kwargs)

        # Construct the inputs of the MessageNode from the set used_nodes
        # TODO simplify this
        if self.node_dict is None:
            inputs = {n.name: n for n in used_nodes}
        else:  # Otherwise we represent inputs as dict
            assert self.node_dict == "auto" or isinstance(self.node_dict, dict)
            spec = inspect.getcallargs(self.fun, *args, **kwargs)  # Read it from the input signature
            if isinstance(self.node_dict, dict):
                spec.update(self.node_dict)
            assert isinstance(spec, dict)
            # Makre sure all nodes in used_nodes are in spec
            assert all([node in spec.values() for node in used_nodes]), "All used_nodes must be in the spec."
            inputs = {k: v for k, v in spec.items() if isinstance(v, Node) and (v in used_nodes)}

        # Wrap the output as a MessageNode
        if self.n_outputs == 1:
            nodes = self.wrap(outputs, inputs)
            parents = set(nodes.parents) if isinstance(nodes, Node) else set()
        else:
            nodes = tuple(self.wrap(outputs[i], inputs) for i in range(self.n_outputs))
            parents = set.union(*[set(node.parents) if isinstance(node, Node) else set() for node in nodes])

        # Make sure all nodes in used_nodes are in the parents of the returned node.
        if nodes is not None and not all([node in parents for node in used_nodes]):
            raise ValueError(
                f"Not all nodes used in the operator {self.fun} are specified as inputs of the returned node."
            )
        return nodes

    def wrap(self, output, inputs: Union[List[Node], Dict[str, Node]]):
        """Wrap the output as a MessageNode of inputs as the parents."""
        if output is None:  # We keep None as None.
            return output
        # Some nodes are used in the operator fun, we need to wrap the output as a MessageNode.
        if not self.wrap_output:  # TODO do we ever use this?
            # If the output is already a Node, we don't need to wrap it.
            # NOTE User who implements fun is responsible for the graph structure.
            if isinstance(output, Node):
                return output
        if self.parameter is not None:
            inputs.update({"__code": self.parameter})
            description = "[eval] This operator eval(__code, *args, **kwargs) evaluates the code block, where __code is the code (str) and *args and **kwargs are the arguments of the function. The output is the result of the evaluation, i.e., __code(*args, **kwargs)."
            name = "eval"
        else:
            description = self.description
            name = self.name
        return MessageNode(output, description=description, inputs=inputs, name=name, info=self.info)

    def __get__(self, obj, objtype):
        # Support instance methods.
        return functools.partial(self.__call__, obj)


if __name__ == "__main__":
    x = node("hello")

    @trace_op("[Custom] This is a test function.")
    def test(x):
        return x.data + " world"

    y = test(x)
    print(y)
    print("Parents", y.parents)
    print("Children", y.children)
    print("Level", y._level)
