
from random import betavariate
from autogen.trace.trace_ops import trace_op
from autogen.trace.nodes import Node

x = Node(1, name='node_x')
y = Node(2, name='node_y')
condition = Node(True)

# Test node_dict=='auto'
@trace_op('[auto_cond] This selects x if condition is True, otherwise y.')
def auto_cond(condition : Node, x : Node, y : Node):
    """
    A function that selects x if condition is True, otherwise y.
    """
    # You can type comments in the function body
    x, y, condition = x, y, condition  # This makes sure all data are read
    return x if condition else y
output = auto_cond(condition, x, y)
assert output.name.split(':')[0] =='auto_cond'
assert output._inputs[x.name] is x and output._inputs[y.name] is y and output._inputs[condition.name] is condition

# Test node_dict=='auto'
# here we use the signature to get the keys of message_node._inputs
@trace_op('[cond] This selects x if condition is True, otherwise y.', node_dict='auto')
def cond(condition : Node, x : Node, y : Node):
    x, y, condition = x, y, condition  # This makes sure all data are read
    return x if condition else y

output = cond(condition, x, y)
assert output.name.split(':')[0] =='cond'
assert output._inputs['x'] is x and output._inputs['y'] is y and output._inputs['condition'] is condition

# Test dot is okay for operator name
@trace_op('[fancy.cond] This selects x if condition is True, otherwise y.', node_dict='auto')
def fancy_cond(condition : Node, x : Node, y : Node):
    x, y, condition = x, y, condition  # This makes sure all data are read
    return x if condition else y
output = fancy_cond(condition, x, y)
assert output.name.split(':')[0] =='fancy.cond'
assert output._inputs['x'] is x and output._inputs['y'] is y and output._inputs['condition'] is condition

# Test wrapping a function that returns a node
@trace_op('[add_1] Add input x and input y')
def foo(x, y):
    z = x + y
    return z
z = foo(x, y)
assert z.data == 3
assert set(z.parents) == {x, y}

# Test tracing class method
class Foo:
    @trace_op('[Foo.add] Add input x and input y')
    def add(self, x, y):
        z = x + y
        return z
foo = Foo()
z = foo.add(x, y)

# Test composition of trace_op with for all_all_methods
from autogen.trace.utils import for_all_methods
@for_all_methods
def test_cls_decorator(fun):
    def wrapper(*args, **kwargs):
        return  fun(*args, **kwargs)
    return wrapper
@test_cls_decorator
class Foo:
    # Test automatic description generation
    @trace_op()
    def add(self, x, y):
        z = x + y
        return z
foo = Foo()
z = foo.add(x, y)
