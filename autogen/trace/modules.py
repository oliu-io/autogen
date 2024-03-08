from autogen.trace.nodes import Node, ParameterNode
import copy


class NodeContainer:
    pass


class Module(NodeContainer):
    # TODO
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def apply_op(op, output, *args, **kwargs):
    """Apply an op to container of Nodes.

    Args:
        op (callable): the operator to be applied.
        output (Any): the container to be updated.
        *args (Any): the positional inputs of the operator.
        **kwargs (Any): the keyword inputs of the operator.
    """

    inputs = list(args) + list(kwargs.values())
    containers = [x for x in inputs if not isinstance(x, Node)]
    if len(containers) == 0:  # all inputs are Nodes, we just apply op
        return op(*args, **kwargs)
    # # there is at least one container
    # output = copy.deepcopy(containers[0])  # this would be used as the template of the output

    def admissible_type(x, base):
        return type(x) == type(base) or isinstance(x, Node)

    assert all(admissible_type(x, output) for x in inputs)  # All inputs are either Nodes or the same type as output

    if isinstance(output, list) or isinstance(output, tuple):
        output = list(output)
        assert all(
            isinstance(x, Node) or len(output) == len(x) for x in inputs
        ), f"output {output} and inputs {inputs} are of different lengths."
        for k in range(len(output)):
            _args = [x if isinstance(x, Node) else x[k] for x in args]
            _kwargs = {kk: vv if isinstance(vv, Node) else vv[k] for kk, vv in kwargs.items()}
            output[k] = apply_op(op, output[k], *_args, **_kwargs)

    elif isinstance(output, dict):
        for k, v in output.items():
            _args = [x if isinstance(x, Node) else x[k] for x in args]
            _kwargs = {kk: vv if isinstance(vv, Node) else vv[k] for kk, vv in kwargs.items()}
            output[k] = apply_op(op, output[k], *_args, **_kwargs)

    elif isinstance(output, NodeContainer):  # this is a NodeContainer object instance
        for k, v in output.__dict__.items():
            _args = [x if isinstance(x, Node) else getattr(x, k) for x in args]
            _kwargs = {kk: vv if isinstance(v, Node) else getattr(vv, k) for kk, vv in kwargs.items()}
            new_v = apply_op(op, v, *_args, **_kwargs)
            setattr(output, k, new_v)
    else:
        pass
    return output


def to_data(obj):
    """Extract the data from a node or a container of nodes."""
    return apply_op(lambda x: x.data, copy.deepcopy(obj), obj)
