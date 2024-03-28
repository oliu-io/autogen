"""
Lisp Interpreter
The ablation part: we specify how complete the 21 functions are. The more complete, the less LLM needs to do.

This requires us to override and implement all the functions required by the Lisp Interpreter.

https://arxiv.org/pdf/2212.10561.pdf

Backup task: Robotic planning

Think about how you can "ablate" this...ok, we write two versions of the function, both are traced
and we just load whichever one we decide.
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

class FullCode:
    @staticmethod
    @trace_op(trainable=True)
    def get_env(parms, args, env=None):
        # Return a new env inside env with parms mapped to their corresponding args, and env as the new env's outer env.
        new_env = {'_outer':env}
        for (parm, arg) in zip(parms, args):
            new_env[parm] = arg
        return new_env

class EmptyCode:
    """
    Can be more barebone than this...
    """
    @staticmethod
    @trace_op(trainable=True)
    def get_env(parms, args, env=None):
        # Return a new env inside env with parms mapped to their corresponding args, and env as the new env's outer env.
        new_env = {'_outer':env}
        return new_env

def bug1():
    GRAPH.clear()

    parms = node([1, 2], trainable=False)
    args = node(['arg1', 'arg2'], trainable=False)

    FullCode.get_env(parms, args)

    # AssertionError: All used_nodes must be in the spec.
    # Sepc values: ["Node: (list:0, dtype=<class 'list'>, data=[1, 2])", "MessageNode: (getitem:0, dtype=<class 'str'>, data=arg1)", "MessageNode: (getitem:1, dtype=<class 'str'>, data=arg2)", 'None']
    # used nodes: ["Node: (list:1, dtype=<class 'list'>, data=['arg1', 'arg2'])", "Node: (list:0, dtype=<class 'list'>, data=[1, 2])"]