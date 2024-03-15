from __future__ import annotations
import trace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # to prevent circular import
    from autogen.trace.nodes import Node
from autogen.trace.trace_ops import trace_op
import copy


@trace_op("[clone] This is a clone operator of x.", node_dict="auto")
def clone(x: Any):
    return copy.deepcopy(x)


def identity(x: Any):
    # identity(x) behaves the same as x.clone()
    return x.clone()


# Unary operators and functions


@trace_op("[pos] This is a pos operator of x.", node_dict="auto")
def pos(x: Any):
    return +x


@trace_op("[neg] This is a neg operator of x.", node_dict="auto")
def neg(x: Any):
    return -x


@trace_op("[abs] This is an abs operator of x.", node_dict="auto")
def abs(x: Any):
    return abs(x)


@trace_op("[invert] This is an invert operator of x.", node_dict="auto")
def invert(x: Any):
    return ~x


@trace_op("[round] This is a round operator of x.", node_dict="auto")
def round(x: Any, n: Any):
    return round(x, n)


@trace_op("[floor] This is a floor operator of x.", node_dict="auto")
def floor(x: Any):
    import math

    return math.floor(x)


@trace_op("[ceil] This is a ceil operator of x.", node_dict="auto")
def ceil(x: Any):
    import math

    return math.ceil(x)


@trace_op("[trunc] This is a trunc operator of x.", node_dict="auto")
def trunc(x: Any):
    import math

    return math.trunc(x)


# Normal arithmetic operators


@trace_op("[add] This is an add operator of x and y.", node_dict="auto")
def add(x: Any, y: Any):
    return x + y


@trace_op("[subtract] This is a subtract operator of x and y.", node_dict="auto")
def subtract(x: Any, y: Any):
    return x - y


@trace_op("[multiply] This is a multiply operator of x and y.", node_dict="auto")
def multiply(x: Any, y: Any):
    return x * y


@trace_op("[floor_divide] This is a floor_divide operator of x and y.", node_dict="auto")
def floor_divide(x: Any, y: Any):
    return x // y


@trace_op("[divide] This is a divide operator of x and y.", node_dict="auto")
def divide(x: Any, y: Any):
    return x / y


@trace_op("[mod] This is a mod operator of x and y.", node_dict="auto")
def mod(x: Any, y: Any):
    return x % y


@trace_op("[divmod] This is a divmod operator of x and y.", node_dict="auto")
def divmod(x: Any, y: Any):
    return divmod(x, y)


@trace_op("[power] This is a power operator of x and y.", node_dict="auto")
def power(x: Any, y: Any):
    return x**y


@trace_op("[lshift] This is a lshift operator of x and y.", node_dict="auto")
def lshift(x: Any, y: Any):
    return x << y


@trace_op("[rshift] This is a rshift operator of x and y.", node_dict="auto")
def rshift(x: Any, y: Any):
    return x >> y


@trace_op("[and] This is an and operator of x and y.", node_dict="auto")
def and_(x: Any, y: Any):
    return x & y


@trace_op("[or] This is an or operator of x and y.", node_dict="auto")
def or_(x: Any, y: Any):
    return x | y


@trace_op("[xor] This is a xor operator of x and y.", node_dict="auto")
def xor(x: Any, y: Any):
    return x ^ y


# Comparison methods


@trace_op("[lt] This is a lt operator of x and y.", node_dict="auto")
def lt(x: Any, y: Any):
    return x < y


@trace_op("[le] This is a le operator of x and y.", node_dict="auto")
def le(x: Any, y: Any):
    return x <= y


@trace_op("[eq] This is an eq operator of x and y.", node_dict="auto")
def eq(x: Any, y: Any):
    return x == y


@trace_op("[ne] This is a ne operator of x and y.", node_dict="auto")
def ne(x: Any, y: Any):
    return x != y


@trace_op("[ge] This is a ge operator of x and y.", node_dict="auto")
def ge(x: Any, y: Any):
    return x >= y


@trace_op("[gt] This is a gt operator of x and y.", node_dict="auto")
def gt(x: Any, y: Any):
    return x > y


# logical operators


@trace_op("[cond] This selects x if condition is True, otherwise y.", node_dict="auto")
def cond(condition: Any, x: Any, y: Any):
    x, y, condition = x, y, condition  # This makes sure all data are read
    return x if condition else y


@trace_op("[not] This is a not operator of x.", node_dict="auto")
def not_(x: Any):
    return not x


@trace_op("[is] Whether x is equal to y.", node_dict="auto")
def is_(x: Any, y: Any):
    return x is y


@trace_op("[is_not] Whether x is not equal to y.", node_dict="auto")
def is_not(x: Any, y: Any):
    return x is not y


@trace_op("[in] Whether x is in y.", node_dict="auto")
def in_(x: Any, y: Any):
    return x in y


@trace_op("[not_in] Whether x is not in y.", node_dict="auto")
def not_in(x: Any, y: Any):
    return x not in y


# Indexing and slicing
@trace_op("[getitem] This is a getitem operator of x based on index.", node_dict="auto")
def getitem(x: Any, index: Any):
    return x[index]


@trace_op("[len] This is a len operator of x.", node_dict="auto")
def len_(x: Any):
    return len(x)

# String operators
@trace_op("[ord] The unicode number of a character.", node_dict="auto")
def ord_(x: Any):
    return ord(x)

@trace_op("[chr] The character of a unicode number.", node_dict="auto")
def chr_(x: Any):
    return chr(x)

@trace_op("[concat] This is a concatenation operator of x and y.", node_dict="auto")
def concat(x: Any, y: Any):
    return x + y

@trace_op("[lower] This makes all characters in x lower case.", node_dict="auto")
def lower(x: Any):
    return x.lower()

@trace_op("[upper] This makes all characters in x upper case.", node_dict="auto")
def upper(x: Any):
    return x.upper()

@trace_op("[title] This makes the first character to upper case and the rest to lower case.", node_dict="auto")
def title(x: Any):
    return x.title()

@trace_op("[swapcase] Swaps the case of all characters: uppercase character to lowercase and viceversa.", node_dict="auto")
def swapcase(x: Any):
    return x.swapcase()

@trace_op("[capitalize] Converts the first character of a string to uppercase.", node_dict="auto")
def capitalize(x: Any):
    return x.capitalize()

@trace_op("[split] Splits the string by finding a substring y in string x, return the first part and second part of string x without y.", node_dict="auto")
def split(x: Any, y: Any, maxsplit: Any = -1):
    return x.split(y, maxsplit)

@trace_op("[strip] Removes the leading and trailing characters of x.", node_dict="auto")
def strip(x: Any):
    return x.strip()

@trace_op("[replace] Replaces all occurrences of substring y in string x with z.", node_dict="auto")
def replace(x: Any, old: Any, new: Any, count: Any = -1):
    return x.replace(old, new, count)