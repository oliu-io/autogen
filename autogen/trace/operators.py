from autogen.trace.trace_operators import trace_operator
from autogen.trace.nodes import Node
import copy

@trace_operator('[cond] This selects x if condition is True, otherwise y.', node_dict='signature')
def cond(condition : Node, x : Node, y : Node):
    x, y, condition = x.data, y.data, condition.data  # This makes sure all data are read
    return x if condition else y

@trace_operator('[add] This is an add operator of x and y.', node_dict='signature')
def add(x : Node, y : Node):
    return x.data + y.data

@trace_operator('[identity] This is an identity operator of x.', node_dict='signature')
def identity(x : Node):
    # identity(x) behaves the same as x.clone()
    return copy.deepcopy(x.data)

@trace_operator('[multiply] This is a multiply operator of x and y.', node_dict='signature')
def multiply(x : Node, y : Node):
    return x.data * y.data

@trace_operator('[divide] This is a divide operator of x and y.', node_dict='signature')
def divide(x : Node, y : Node):
    return x.data / y.data

@trace_operator('[subtract] This is a subtract operator of x and y.', node_dict='signature')
def subtract(x : Node, y : Node):
    return x.data - y.data

@trace_operator('[power] This is a power operator of x and y.', node_dict='signature')
def power(x : Node, y : Node):
    return x.data ** y.data

@trace_operator('[mod] This is a mod operator of x and y.', node_dict='signature')
def mod(x : Node, y : Node):
    return x.data % y.data

@trace_operator('[floor_divide] This is a floor_divide operator of x and y.', node_dict='signature')
def floor_divide(x : Node, y : Node):
    return x.data // y.data

@trace_operator('[abs] This is an abs operator of x.', node_dict='signature')
def abs(x : Node):
    return abs(x.data)

@trace_operator('[neg] This is a neg operator of x.', node_dict='signature')
def neg(x : Node):
    return -x.data

@trace_operator('[pos] This is a pos operator of x.', node_dict='signature')
def pos(x : Node):
    return +x.data

@trace_operator('[invert] This is an invert operator of x.', node_dict='signature')
def invert(x : Node):
    return ~x.data

@trace_operator('[lshift] This is a lshift operator of x and y.', node_dict='signature')
def lshift(x : Node, y : Node):
    return x.data << y.data

@trace_operator('[rshift] This is a rshift operator of x and y.', node_dict='signature')
def rshift(x : Node, y : Node):
    return x.data >> y.data

@trace_operator('[and] This is an and operator of x and y.', node_dict='signature')
def and_(x : Node, y : Node):
    return x.data & y.data

@trace_operator('[or] This is an or operator of x and y.', node_dict='signature')
def or_(x : Node, y : Node):
    return x.data | y.data

@trace_operator('[xor] This is a xor operator of x and y.', node_dict='signature')
def xor(x : Node, y : Node):
    return x.data ^ y.data

@trace_operator('[lt] This is a lt operator of x and y.', node_dict='signature')
def lt(x : Node, y : Node):
    return x.data < y.data

@trace_operator('[le] This is a le operator of x and y.', node_dict='signature')
def le(x : Node, y : Node):
    return x.data <= y.data

@trace_operator('[eq] This is an eq operator of x and y.', node_dict='signature')
def eq(x : Node, y : Node):
    return x.data == y.data

@trace_operator('[ne] This is a ne operator of x and y.', node_dict='signature')
def ne(x : Node, y : Node):
    return x.data != y.data

@trace_operator('[ge] This is a ge operator of x and y.', node_dict='signature')
def ge(x : Node, y : Node):
    return x.data >= y.data

@trace_operator('[gt] This is a gt operator of x and y.', node_dict='signature')
def gt(x : Node, y : Node):
    return x.data > y.data

@trace_operator('[not] This is a not operator of x.', node_dict='signature')
def not_(x : Node):
    return not x.data

@trace_operator('[is] This is an is operator of x and y.', node_dict='signature')
def is_(x : Node, y : Node):
    return x.data is y.data

@trace_operator('[is_not] This is an is_not operator of x and y.', node_dict='signature')
def is_not(x : Node, y : Node):
    return x.data is not y.data

@trace_operator('[in] This is an in operator of x and y.', node_dict='signature')
def in_(x : Node, y : Node):
    return x.data in y.data

@trace_operator('[not_in] This is a not_in operator of x and y.', node_dict='signature')
def not_in(x : Node, y : Node):
    return x.data not in y.data

@trace_operator('[getitem] This is a getitem operator of x based on index.', node_dict='signature')
def getitem(x : Node, index : Node):
    return x.data[index.data]

@trace_operator('[len] This is a len operator of x.', node_dict='signature')
def len_(x : Node):
    return len(x.data)

@trace_operator('[contains] This is a contains operator of x and y.', node_dict='signature')
def contains(x : Node, y : Node):
    return y.data in x.data
