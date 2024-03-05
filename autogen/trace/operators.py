from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # to prevent cicular import
    from autogen.trace.nodes import Node
from autogen.trace.trace_ops import trace_op
import copy

@trace_op('[cond] This selects x if condition is True, otherwise y.', node_dict='signature')
def cond(condition : Node, x : Node, y : Node):
    x, y, condition = x.data, y.data, condition.data  # This makes sure all data are read
    return x if condition else y

@trace_op('[add] This is an add operator of x and y.', node_dict='signature')
def add(x : Node, y : Node):
    return x.data + y.data

@trace_op('[identity] This is an identity operator of x.', node_dict='signature')
def identity(x : Node):
    # identity(x) behaves the same as x.clone()
    return copy.deepcopy(x.data)

@trace_op('[multiply] This is a multiply operator of x and y.', node_dict='signature')
def multiply(x : Node, y : Node):
    return x.data * y.data

@trace_op('[divide] This is a divide operator of x and y.', node_dict='signature')
def divide(x : Node, y : Node):
    return x.data / y.data

@trace_op('[subtract] This is a subtract operator of x and y.', node_dict='signature')
def subtract(x : Node, y : Node):
    return x.data - y.data

@trace_op('[power] This is a power operator of x and y.', node_dict='signature')
def power(x : Node, y : Node):
    return x.data ** y.data

@trace_op('[mod] This is a mod operator of x and y.', node_dict='signature')
def mod(x : Node, y : Node):
    return x.data % y.data

@trace_op('[floor_divide] This is a floor_divide operator of x and y.', node_dict='signature')
def floor_divide(x : Node, y : Node):
    return x.data // y.data

@trace_op('[neg] This is a neg operator of x.', node_dict='signature')
def neg(x : Node):
    return -x.data

@trace_op('[pos] This is a pos operator of x.', node_dict='signature')
def pos(x : Node):
    return +x.data

@trace_op('[invert] This is an invert operator of x.', node_dict='signature')
def invert(x : Node):
    return ~x.data

@trace_op('[lshift] This is a lshift operator of x and y.', node_dict='signature')
def lshift(x : Node, y : Node):
    return x.data << y.data

@trace_op('[rshift] This is a rshift operator of x and y.', node_dict='signature')
def rshift(x : Node, y : Node):
    return x.data >> y.data

@trace_op('[and] This is an and operator of x and y.', node_dict='signature')
def and_(x : Node, y : Node):
    return x.data & y.data

@trace_op('[or] This is an or operator of x and y.', node_dict='signature')
def or_(x : Node, y : Node):
    return x.data | y.data

@trace_op('[xor] This is a xor operator of x and y.', node_dict='signature')
def xor(x : Node, y : Node):
    return x.data ^ y.data

@trace_op('[lt] This is a lt operator of x and y.', node_dict='signature')
def lt(x : Node, y : Node):
    return x.data < y.data

@trace_op('[le] This is a le operator of x and y.', node_dict='signature')
def le(x : Node, y : Node):
    return x.data <= y.data

@trace_op('[eq] This is an eq operator of x and y.', node_dict='signature')
def eq(x : Node, y : Node):
    return x.data == y.data

@trace_op('[ne] This is a ne operator of x and y.', node_dict='signature')
def ne(x : Node, y : Node):
    return x.data != y.data

@trace_op('[ge] This is a ge operator of x and y.', node_dict='signature')
def ge(x : Node, y : Node):
    return x.data >= y.data

@trace_op('[gt] This is a gt operator of x and y.', node_dict='signature')
def gt(x : Node, y : Node):
    return x.data > y.data

@trace_op('[not] This is a not operator of x.', node_dict='signature')
def not_(x : Node):
    return not x.data

@trace_op('[is] This is an is operator of x and y.', node_dict='signature')
def is_(x : Node, y : Node):
    return x.data is y.data

@trace_op('[is_not] This is an is_not operator of x and y.', node_dict='signature')
def is_not(x : Node, y : Node):
    return x.data is not y.data

@trace_op('[in] This is an in operator of x and y.', node_dict='signature')
def in_(x : Node, y : Node):
    return x.data in y.data

@trace_op('[not_in] This is a not_in operator of x and y.', node_dict='signature')
def not_in(x : Node, y : Node):
    return x.data not in y.data

@trace_op('[getitem] This is a getitem operator of x based on index.', node_dict='signature')
def getitem(x : Node, index : Node):
    return x.data[index.data]

@trace_op('[len] This is a len operator of x.', node_dict='signature')
def len_(x : Node):
    return len(x.data)

@trace_op('[contains] This is a contains operator of x and y.', node_dict='signature')
def contains(x : Node, y : Node):
    return y.data in x.data
