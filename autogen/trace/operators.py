from __future__ import annotations
import trace
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # to prevent cicular import
    from autogen.trace.nodes import Node
from autogen.trace.trace_ops import trace_op
import copy


def identity(x : Node):
    # identity(x) behaves the same as x.clone()
    return x.clone()

# Unary operators and functions

@trace_op('[pos] This is a pos operator of x.', node_dict='auto')
def pos(x : Node):
    return +x.data

@trace_op('[neg] This is a neg operator of x.', node_dict='auto')
def neg(x : Node):
    return -x.data

@trace_op('[abs] This is an abs operator of x.', node_dict='auto')
def abs(x : Node):
    return abs(x.data)

@trace_op('[invert] This is an invert operator of x.', node_dict='auto')
def invert(x : Node):
    return ~x.data

@trace_op('[round] This is a round operator of x.', node_dict='auto')
def round(x : Node, n : Node):
    return round(x.data, n.data)

@trace_op('[floor] This is a floor operator of x.', node_dict='auto')
def floor(x : Node):
    import math
    return math.floor(x.data)

@trace_op('[ceil] This is a ceil operator of x.', node_dict='auto')
def ceil(x : Node):
    import math
    return math.ceil(x.data)

@trace_op('[trunc] This is a trunc operator of x.', node_dict='auto')
def trunc(x : Node):
    import math
    return math.trunc(x.data)

# Normal arithmetic operators

@trace_op('[add] This is an add operator of x and y.', node_dict='auto')
def add(x : Node, y : Node):
    return x.data + y.data

@trace_op('[subtract] This is a subtract operator of x and y.', node_dict='auto')
def subtract(x : Node, y : Node):
    return x.data - y.data

@trace_op('[multiply] This is a multiply operator of x and y.', node_dict='auto')
def multiply(x : Node, y : Node):
    return x.data * y.data

@trace_op('[floor_divide] This is a floor_divide operator of x and y.', node_dict='auto')
def floor_divide(x : Node, y : Node):
    return x.data // y.data

@trace_op('[divide] This is a divide operator of x and y.', node_dict='auto')
def divide(x : Node, y : Node):
    return x.data / y.data

@trace_op('[mod] This is a mod operator of x and y.', node_dict='auto')
def mod(x : Node, y : Node):
    return x.data % y.data

@trace_op('[divmod] This is a divmod operator of x and y.', node_dict='auto')
def divmod(x : Node, y : Node):
    return divmod(x.data, y.data)

@trace_op('[power] This is a power operator of x and y.', node_dict='auto')
def power(x : Node, y : Node):
    return x.data ** y.data

@trace_op('[lshift] This is a lshift operator of x and y.', node_dict='auto')
def lshift(x : Node, y : Node):
    return x.data << y.data

@trace_op('[rshift] This is a rshift operator of x and y.', node_dict='auto')
def rshift(x : Node, y : Node):
    return x.data >> y.data

@trace_op('[and] This is an and operator of x and y.', node_dict='auto')
def and_(x : Node, y : Node):
    return x.data & y.data

@trace_op('[or] This is an or operator of x and y.', node_dict='auto')
def or_(x : Node, y : Node):
    return x.data | y.data

@trace_op('[xor] This is a xor operator of x and y.', node_dict='auto')
def xor(x : Node, y : Node):
    return x.data ^ y.data

# Comparison methods

@trace_op('[lt] This is a lt operator of x and y.', node_dict='auto')
def lt(x : Node, y : Node):
    return x.data < y.data

@trace_op('[le] This is a le operator of x and y.', node_dict='auto')
def le(x : Node, y : Node):
    return x.data <= y.data

@trace_op('[eq] This is an eq operator of x and y.', node_dict='auto')
def eq(x : Node, y : Node):
    return x.data == y.data

@trace_op('[ne] This is a ne operator of x and y.', node_dict='auto')
def ne(x : Node, y : Node):
    return x.data != y.data

@trace_op('[ge] This is a ge operator of x and y.', node_dict='auto')
def ge(x : Node, y : Node):
    return x.data >= y.data

@trace_op('[gt] This is a gt operator of x and y.', node_dict='auto')
def gt(x : Node, y : Node):
    return x.data > y.data

# logical operators

@trace_op('[cond] This selects x if condition is True, otherwise y.', node_dict='auto')
def cond(condition : Node, x : Node, y : Node):
    x, y, condition = x.data, y.data, condition.data  # This makes sure all data are read
    return x if condition else y

@trace_op('[not] This is a not operator of x.', node_dict='auto')
def not_(x : Node):
    return not x.data

@trace_op('[is] Whether x is equal to y.', node_dict='auto')
def is_(x : Node, y : Node):
    return x.data is y.data

@trace_op('[is_not] Whether x is not equal to y.', node_dict='auto')
def is_not(x : Node, y : Node):
    return x.data is not y.data

@trace_op('[in] Whether x is in y.', node_dict='auto')
def in_(x : Node, y : Node):
    return x.data in y.data

@trace_op('[not_in] Whether x is not in y.', node_dict='auto')
def not_in(x : Node, y : Node):
    return x.data not in y.data

# Indexing and slicing
@trace_op('[getitem] This is a getitem operator of x based on index.', node_dict='auto')
def getitem(x : Node, index : Node, node_dict='auto'):
    return x.data[index.data]

@trace_op('[len] This is a len operator of x.', node_dict='auto')
def len_(x : Node):
    return len(x.data)
