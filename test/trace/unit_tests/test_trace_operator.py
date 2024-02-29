
from autogen.trace.trace_operators import trace_operator
from autogen.trace.nodes import Node

x = Node(1, name='node_x')
y = Node(2, name='node_y')
condition = Node(True)

# Test node_dict=='auto'
@trace_operator('[auto_cond] This selects x if condition is True, otherwise y.')
def auto_cond(condition : Node, x : Node, y : Node):
    x, y, condition = x.data, y.data, condition.data  # This makes sure all data are read
    return x if condition else y
output = auto_cond(condition, x, y)
assert output.name.split(':')[0] =='auto_cond'
assert output._inputs[x.name] is x and output._inputs[y.name] is y and output._inputs[condition.name] is condition

# Test node_dict=='signature'
# here we use the signature to get the keys of message_node._inputs
@trace_operator('[cond] This selects x if condition is True, otherwise y.', node_dict='signature')
def cond(condition : Node, x : Node, y : Node):
    x, y, condition = x.data, y.data, condition.data  # This makes sure all data are read
    return x if condition else y

output = cond(condition, x, y)
assert output.name.split(':')[0] =='cond'
assert output._inputs['x'] is x and output._inputs['y'] is y and output._inputs['condition'] is condition

# Test dot is okay for operator name
@trace_operator('[fancy.cond] This selects x if condition is True, otherwise y.', node_dict='signature')
def fancy_cond(condition : Node, x : Node, y : Node):
    x, y, condition = x.data, y.data, condition.data  # This makes sure all data are read
    return x if condition else y
output = fancy_cond(condition, x, y)
assert output.name.split(':')[0] =='fancy.cond'
assert output._inputs['x'] is x and output._inputs['y'] is y and output._inputs['condition'] is condition
