from autogen.trace.nodes import node, GRAPH
import string
import random
import numpy as np
from textwrap import dedent

from typing import List
import copy
from autogen.trace.trace_ops import FunModule, trace_op

import io, contextlib
import sys
import time
import resource
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq

distinct_functions = {
    "evaluate_program",
    "get_env",
    "standard_env",
    "get_math",
    "get_ops",
    "get_simple_math",
    "apply_fn_dict_key",
    "parse_and_update",
    "eval_exp",
    "find",
    "string_case",
    "list_case",
    "get_procedure",
    "eval_procedure",
    "otherwise_case",
    "not_list_case",
    "parse",
    "tokenize",
    "read_from_tokens",
    "atom",
    "nested_list_to_str",
}

"""
We will let Trace implement low level atomic functions, but high-level connections are defined by people

We still use @random_trace so that we can vary between how many functions are used

TODO:
1. Write the feedback provider / unit test (how to do this??)
"""


# we can use this function to design the task
# to offer control of difficulty and complexity
def get_VERSION():
    # Randomly returns 'full' or 'empty' to simulate dynamic function behavior
    # return "full" if random.random() > 0.5 else "empty"
    return 'full'


def random_trace(description=None, n_outputs=1, node_dict="auto",
                 wrap_output=True, unpack_input=True, trainable=False,
                 allow_external_dependencies=False, catch_execution_error=True):
    def decorator(fun):
        if get_VERSION() == "full":
            return FunModule(
                fun=fun,
                description=description,
                n_outputs=n_outputs,
                node_dict=node_dict,
                wrap_output=wrap_output,
                unpack_input=unpack_input,
                trainable=trainable,
                decorator_name="@random_trace",
                allow_external_dependencies=allow_external_dependencies,
                catch_execution_error=catch_execution_error
            )
        else:
            return FunModule(
                fun=getattr(EmptyFuncs, fun.__name__),
                description=description,
                n_outputs=n_outputs,
                node_dict=node_dict,
                wrap_output=wrap_output,
                unpack_input=unpack_input,
                trainable=True,
                decorator_name="",
                allow_external_dependencies=allow_external_dependencies,
                catch_execution_error=catch_execution_error
            )

    return decorator

"""
We put empty functions first
"""

class EmptyFuncs:
    @staticmethod
    def get_env(parms, args, env=None):
        """
        [], [] -> {'_outer': None}
        ['a'], [1] -> {'a': 1, '_outer': None}
        """
        new_env = {}
        return new_env

    @staticmethod
    def get_math():
        """
        'sqrt', [4] -> 2.0
        """
        d = {}
        return d

    @staticmethod
    def get_ops():
        """
        '+', [1, 2] -> 3
        """
        return {}

    @staticmethod
    def get_simple_math():
        """
        'abs', [-1] -> 1
        Calls:
        - get_math
        - get_ops
        - get_simple_math
        """
        return {}

    @staticmethod
    def apply_fn_dict_key(fn_dict_generator, key, args_list):
        return 0

    @staticmethod
    def string_case(x, env):
        """
        'a', {'a':4, '_outer':None} -> 4
        """
        return 4

    @staticmethod
    def eval_procedure(parms, body, env, args):
        """
        ['r'], ['*', 'pi', ['*', 'r', 'r']], {'*': (lambda x, y: x * y), 'pi': 3, '_outer': None}, [1] -> 3
        """
        return

    @staticmethod
    def otherwise_case(x, env):
        """
        ['+', 1, 2], {'+': (lambda x, y: x + y), '_outer': None} -> 3
        """
        return 3


def get_env(parms, args, env=None):
    new_env = {"_outer": env}
    for parm, arg in zip(parms, args):
        new_env[parm] = arg
    return new_env


def get_math():
    d = {}
    for name in dir(math):
        if name[:2] != "__":
            d[name] = getattr(math, name)
    return d


def get_ops():
    return {
        "+": (lambda x, y: x + y),
        "-": (lambda x, y: x - y),
        "*": (lambda x, y: x * y),
        "/": (lambda x, y: x / y),
        ">": (lambda x, y: x > y),
        "<": (lambda x, y: x < y),
        ">=": (lambda x, y: x >= y),
        "<=": (lambda x, y: x <= y),
        "=": (lambda x, y: x == y),
    }


def get_simple_math():
    return {"abs": abs, "min": min, "max": max, "not": lambda x: not x, "round": round}


def apply_fn_dict_key(fn_dict_generator, key, args_list):
    fn_dict = fn_dict_generator()
    return fn_dict[key](*args_list)


def find(env, var):
    if var in env:
        return env[var]
    else:
        return find(env["_outer"], var)


def string_case(x, env):
    return find(env, x)


def eval_procedure(parms, body, env, args):
    get_procedure(parms, body, env)
    new_env = get_env(parms, args, env)
    return eval_exp(body, new_env)


def get_procedure(parms, body, env):
    return lambda *args: eval_procedure(parms, body, env, args)


@random_trace(
    description="[otherwise_case] Get the procedure by evaluating the first value of x. Then, evaluate the arguments and apply the procedure to them. Return the result.",
    allow_external_dependencies=True
)
def otherwise_case(x, env):
    p = eval_exp(x[0], env)
    args = [eval_exp(arg, env) for arg in x[1:]]
    return p(*args)

@random_trace(
    description="[list_case] Handle the function specified by the first value of x. Handle the first value of x being quote, if, define, set!, lambda, or otherwise. Return the result.",
    allow_external_dependencies=True,
    unpack_input=True,
    catch_execution_error=False
)
def list_case(x, env):
    if x[0] == 'quote':
        return x[1]
    elif x[0] == 'if':
        if eval_exp(x[1], env):
            return eval_exp(x[2], env)
        elif len(x) == 4:
            return eval_exp(x[3], env)
    elif x[0] == 'define':
        env[x[1]] = eval_exp(x[2], env)
        print("inside define", env)
    elif x[0] == 'set!':
        env.find(x[1])[x[1]] = eval_exp(x[2], env)
    elif x[0] == 'lambda':
        return get_procedure(x[1], x[2], env)
    else:
        proc = eval_exp(x[0], env)
        args = [ eval_exp(arg, env) for arg in x[1:] ]
        return proc(*args)

@random_trace(
    description="[not_list_case] Return x if it's not a list, otherwise return None.",
    allow_external_dependencies=True
)
def not_list_case(x, env):
    if isinstance(x, list):
        return None
    return x

@trace_op(description="[eval_exp] Evaluate an expression in an environment and return the result. Check if x is a list, a string, or neither, and call the corresponding function.",
         allow_external_dependencies=True, catch_execution_error=False, unpack_input=True)
def eval_exp(x, env):
    print("inside eval_exp", env)
    if isinstance(x, list):
        return list_case(x, env)
    elif isinstance(x, str):
        return string_case(x, env)
    else:
        return not_list_case(x, env)


@random_trace(description="[tokenize] Convert a string into a list of tokens, including parens.")
def tokenize(s):
    "Convert a string into a list of tokens, including parens."
    return s.replace("(", " ( ").replace(")", " ) ").split()

@trace_op(description="[atom] Infer type of a token")
def atom(token):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


def read_from_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError("unexpected EOF while reading")
    token = tokens.pop(0)
    if token == "(":
        L = []
        while tokens[0] != ")":
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif token == ")":
        raise SyntaxError("unexpected )")
    else:
        return atom(token)


def parse(program):
    return read_from_tokens(tokenize(program))


@trace_op(description="[nested_list_to_str] Convert a nested list into a string with nesting represented by parentheses.")
def nested_list_to_str(exp):
    if isinstance(exp, list):
        return '(' + ' '.join(map(nested_list_to_str, exp)) + ')'
    else:
        return str(exp)


def parse_and_update(expression, env):
    exp = parse(expression)
    result = eval_exp(exp, env)
    return nested_list_to_str(result)


def standard_env(includes=["math", "ops", "simple_math"]):
    env = {"_outer": None}
    if "math" in includes:
        env.update(get_math())
    if "ops" in includes:
        env.update(get_ops())
    if "simple_math" in includes:
        env.update(get_simple_math())
    return env


# Initialize a standard environment. Parse and evaluate a list of expressions, returning the final result.
def evaluate_program(program):
    env = node(standard_env())
    last = None
    for expression in program:
        last = parse_and_update(expression, env)
    return last

if __name__ == '__main__':
    env = {'_outer': None}
    expression = '(define square (lambda (r) (* r r)))'
    exp = parse(expression)
    result = eval_exp(exp, env)
    print(env)