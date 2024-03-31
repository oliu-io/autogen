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
