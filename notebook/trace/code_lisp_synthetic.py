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
    """

    @staticmethod
    def get_env(parms, args, env=None):
        new_env = {'_outer': env}
        return new_env

    @staticmethod
    def get_math():
        d = {}
        return d

    @staticmethod
    def get_ops():
        return {}

    @staticmethod
    def get_simple_math():
        return {}

    @staticmethod
    def apply_fn_dict_key(fn_dict_generator, key, args_list):
        return

    @staticmethod
    def standard_env(includes=['math', 'ops', 'simple_math']):
        env = {'_outer': None}
        return env

    @staticmethod
    def find(env, var):
        return

    @staticmethod
    def string_case(x, env):
        return


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
    description="[standard_env] An environment with some Scheme standard procedures. Start with an environment and update it with standard functions.")
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
