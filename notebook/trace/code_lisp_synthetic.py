"""
Lisp Interpreter
The ablation part: we specify how complete the 21 functions are. The more complete, the less LLM needs to do.

This requires us to override and implement all the functions required by the Lisp Interpreter.

https://arxiv.org/pdf/2212.10561.pdf

Backup task: Robotic planning

Think about how you can "ablate" this...ok, we write two versions of the function, both are traced
and we just load whichever one we decide.

Original Parsel needs to backtrack and do error trace, and few-shot demonstration
We are making this task harder
"""

from autogen.trace.nodes import node, GRAPH
from autogen.trace.propagators import FunctionPropagator
import string
import random
import numpy as np
from textwrap import dedent

from typing import List
import copy
from autogen.trace.operators import *
from autogen.trace.trace_ops import FunModule

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
    "evaluate_program", "get_env", "standard_env", "get_math", "get_ops",
    "get_simple_math", "apply_fn_dict_key", "parse_and_update", "eval_exp",
    "find", "string_case", "list_case", "get_procedure", "eval_procedure",
    "otherwise_case", "not_list_case", "parse", "tokenize", "read_from_tokens",
    "atom", "nested_list_to_str"
}


# we can use this function to design the task
# to offer control of difficulty and complexity
def get_VERSION():
    # Randomly returns 'full' or 'empty' to simulate dynamic function behavior
    return 'full' if random.random() > 0.5 else 'empty'


def random_trace(description=None, n_outputs=1, node_dict="auto", wrap_output=True, unpack_input=True, trainable=False):
    def decorator(fun):
        if get_VERSION() == 'full':
            return FunModule(
                fun=fun,
                description=description,
                n_outputs=n_outputs,
                node_dict=node_dict,
                wrap_output=wrap_output,
                unpack_input=unpack_input,
                trainable=trainable,
                decorator_name="@random_trace"
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
                decorator_name=""
            )

    return decorator


class EmptyFuncs:
    """
    All the empty functions specify the inputs and outputs correctly.
    Also, make sure the empty version is still runnable, just won't produce the incorrect result

    Test a version where it throws error/exception and see how you want to handle this.

    Let's write the input/output according to Parsel's definition
    """

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
    def standard_env(includes=['math', 'ops', 'simple_math']):
        """
        [] -> {'_outer': None}
        """
        env = {'_outer': None}
        return env

    @staticmethod
    def find(env, var):
        """
        {'a':4, '_outer':None}, 'a' -> 4
        {'_outer':{'a':4, '_outer':None}}, 'a' -> 4
        {'a':3, '_outer':{'a':4, '_outer':None}}, 'a' -> 3
        """
        return 4

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
    def get_procedure(parms, body, env):
        """
        Calls eval_procedure
        """
        return lambda *args: 3

    @staticmethod
    def otherwise_case(x, env):
        """
        ['+', 1, 2], {'+': (lambda x, y: x + y), '_outer': None} -> 3
        """
        return 3

    @staticmethod
    def list_case(x, env):
        """
        ['quote', 'a'], {'_outer': None} -> 'a'
        ['if', True, 1, 2], {'_outer': None} -> 1
        ['define', 'a', 1], {'_outer': None} -> None
        """
        return 1

    @staticmethod
    def not_list_case(x, env):
        """
        1, {} -> 1
        """
        return 1

    @staticmethod
    def eval_exp(x, env):
        """
        1, {'_outer': None} -> 1
        """
        return 1

    @staticmethod
    def tokenize(s):
        """
        "1 + 2" -> ['1', '+', '2']
        "1 + (2 * 3)" -> ['1', '+', '(', '2', '*', '3', ')']
        """
        return ['1', '+', '(', '2', '*', '3', ')']

    @staticmethod
    def atom(token):
        """
        "1" -> 1
        "a" -> "a"
        "1.2" -> 1.2
        """
        return 1

    @staticmethod
    def read_from_tokens(tokens):
        """
        ['(', '1', '+', '(', '2', '*', '3', ')', ')'] -> [1, '+', [2, '*', 3]]
        """
        return [1, '+', [2, '*', 3]]

    @staticmethod
    def parse(program):
        """
        '(1 + (2 * 3))' -> [1, '+', [2, '*', 3]]
        """
        return [1, '+', [2, '*', 3]]

    @staticmethod
    def nested_list_to_str(exp):
        """
        1 -> "1"
        [1, '+', [2, '*', 3]] -> "(1 + (2 * 3))"
        """
        return "(1 + (2 * 3))"

    @staticmethod
    def parse_and_update(expression, env):
        """
        "(+ 1 (* 2 3))", {'+': (lambda x, y: x + y), '*': (lambda x, y: x * y), '_outer': None} -> 7
        """
        return 7


@random_trace(
    description="[get_env] Return a new env inside env with parms mapped to their corresponding args, and env as the new env's outer env.")
def get_env(parms, args, env=None):
    new_env = {'_outer': env}
    for (parm, arg) in zip(parms, args):
        new_env[parm] = arg
    return new_env


@random_trace(description="[get_math] Get a dictionary mapping math library function names to their functions.")
def get_math():
    d = {}
    for name in dir(math):
        if name[:2] != '__':
            d[name] = getattr(math, name)
    return d


@random_trace(description="[get_ops] Get a dictionary mapping math library function names to their functions.")
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
        "=": (lambda x, y: x == y)
    }


@random_trace(
    description="[get_simple_math] Get a dictionary mapping 'abs', 'min', 'max', 'not', 'round' to their functions.")
def get_simple_math():
    return {'abs': abs, 'min': min, 'max': max, 'not': lambda x: not x, 'round': round}


@random_trace(
    description="[apply_fn_dict_key] Return the value of fn_dict_generator()[key](*args_list) in standard_env.")
def apply_fn_dict_key(fn_dict_generator, key, args_list):
    fn_dict = fn_dict_generator()
    return fn_dict[key](*args_list)


@random_trace(
    description="[standard_env] An environment with some Scheme standard procedures. Start with an environment and update it with standard functions.",
    node_dict=None)
def standard_env(includes=['math', 'ops', 'simple_math']):
    env = {'_outer': None}
    if 'math' in includes:
        env.update(get_math())
    if 'ops' in includes:
        env.update(get_ops())
    if 'simple_math' in includes:
        env.update(get_simple_math())
    return env


@random_trace(description="[find] Find the value of var in the innermost env where var appears.")
def find(env, var):
    if var in env:
        return env[var]
    else:
        return find(env['_outer'], var)


@random_trace(description="[string_case] Return find(env, x).")
def string_case(x, env):
    return find(env, x)


@random_trace(
    description="[eval_procedure] Gets a procedure and returns the result of evaluating proc(*args) in env. Should not be called directly.")
def eval_procedure(parms, body, env, args):
    proc = get_procedure(parms, body, env)
    new_env = get_env(parms, args, env)
    return eval_exp(body, new_env)


@random_trace(
    description="[get_procedure] Return a procedure which evaluates body in a new environment with parms bound to the args passed to the procedure (in the same order as parms).")
def get_procedure(parms, body, env):
    return lambda *args: eval_procedure(parms, body, env, args)


@random_trace(
    description="[otherwise_case] Get the procedure by evaluating the first value of x. Then, evaluate the arguments and apply the procedure to them. Return the result.")
def otherwise_case(x, env):
    p = eval_exp(x[0], env)
    args = [eval_exp(arg, env) for arg in x[1:]]
    return p(*args)


@random_trace(
    description="[list_case] Handle the function specified by the first value of x. Handle the first value of x being quote, if, define, set!, lambda, or otherwise. Return the result.")
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
    elif x[0] == 'set!':
        env.find(x[1])[x[1]] = eval_exp(x[2], env)
    elif x[0] == 'lambda':
        return get_procedure(x[1], x[2], env)
    else:
        proc = eval_exp(x[0], env)
        args = [eval_exp(arg, env) for arg in x[1:]]
        return proc(*args)


@random_trace(description="[not_list_case] Return x.")
def not_list_case(x, env):
    if isinstance(x, list):
        return None
    return x


@random_trace(
    description="[eval_exp] Evaluate an expression in an environment and return the result. Check if x is a list, a string, or neither, and call the corresponding function.")
def eval_exp(x, env):
    if isinstance(x, list):
        return list_case(x, env)
    elif isinstance(x, str):
        return string_case(x, env)
    else:
        return not_list_case(x, env)


@random_trace(description="[tokenize] Convert a string into a list of tokens, including parens.")
def tokenize(s):
    "Convert a string into a list of tokens, including parens."
    return s.replace('(', ' ( ').replace(')', ' ) ').split()


@random_trace(description="[atom] Numbers become numbers; every other token is a string.")
def atom(token):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


@random_trace(
    description="[read_from_tokens] Translate tokens to their corresponding atoms, using parentheses for nesting lists.")
def read_from_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        return atom(token)


@random_trace(description="[parse] Read a Scheme expression from a string.")
def parse(program):
    return read_from_tokens(tokenize(program))


@random_trace(
    description="[nested_list_to_str] Convert a nested list into a string with nesting represented by parentheses.")
def nested_list_to_str(exp):
    if isinstance(exp, list):
        return '(' + ' '.join(map(nested_list_to_str, exp)) + ')'
    else:
        return str(exp)


@random_trace(description="[parse_and_update] Parse an expression, return the result.")
def parse_and_update(expression, env):
    exp = parse(expression)
    result = eval_exp(exp, env)
    return nested_list_to_str(result)


# Initialize a standard environment. Parse and evaluate a list of expressions, returning the final result.
def evaluate_program(program):
    """
    ['(define square (lambda (r) (* r r)))', '(square 3)'] -> 9
    """
    env = standard_env()
    last = None
    for expression in program:
        last = parse_and_update(expression, env)
    return last
