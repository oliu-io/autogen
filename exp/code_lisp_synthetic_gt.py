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
from autogen.trace.trace_ops import FunModule, trace_op, trace_class
from autogen.trace.nodes import Node
import math
import random

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


def standard_env(includes=["math", "ops", "simple_math"]):
    env = {"_outer": None}
    if "math" in includes:
        env.update(get_math())
    if "ops" in includes:
        env.update(get_ops())
    if "simple_math" in includes:
        env.update(get_simple_math())
    return env


class Environment(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."

    def __init__(self, parms=(), args=(), outer=None):
        for p, a in zip(parms, args):
            if isinstance(a, Node):
                a = a.data
            if isinstance(p, Node):
                p = p.data
            self[p] = a

        if isinstance(outer, Node):
            outer = outer.data

        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        if isinstance(var, Node):
            var = var.data
        return self if (var in self) else self.outer.find(var)

    def __setitem__(self, key, value):
        if isinstance(value, Node):
            value = value.data
        if isinstance(key, Node):
            key = key.data
        super().__setitem__(key, value)

# what if we use this to showcase multi-agent conversation using this...
# we write a tester class/function that can test the output of the program, provide feedback dynamically
# multi-round conversation

"""
1. Interpreter itself cannot be changed/updated...

A typical work flow in coding is this writer-test thing (unit test generation)
(But that's kinda hard to control)
We can do this with trace easily

We would write a function/test-function pair
then in the outer loop (env/feedback), call both?

function -> self-test -> output -> get feedback, update both self-test and function
(two optimizers, adversarial style)

self-test gets reward -1 if the model passes the self-test and then fail the hidden test
self-test gets reward +1 if the model fails and also fails the hidden test
We ignore the case where self-test fails the model but the model passes the hidden test (because we terminate already)

We are doing adversarial training.
"""
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


def standard_env(includes=["math", "ops", "simple_math"]):
    env = {"_outer": None}
    if "math" in includes:
        env.update(get_math())
    if "ops" in includes:
        env.update(get_ops())
    if "simple_math" in includes:
        env.update(get_simple_math())
    return env


class Environment(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."

    def __init__(self, parms=(), args=(), outer=None):
        for p, a in zip(parms, args):
            if isinstance(a, Node):
                a = a.data
            if isinstance(p, Node):
                p = p.data
            self[p] = a

        if isinstance(outer, Node):
            outer = outer.data

        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        if isinstance(var, Node):
            var = var.data
        return self if (var in self) else self.outer.find(var)

    def __setitem__(self, key, value):
        if isinstance(value, Node):
            value = value.data
        if isinstance(key, Node):
            key = key.data
        super().__setitem__(key, value)


global_env = Environment()
global_env.update(standard_env())

@trace_class
class CodingAgent:

    @trace_op(description="[atom] Numbers become numbers; every other token is a symbol.",
              trainable=True)
    def atom(self, token):
        return token

    @trace_op(
        description="[test_atom] Test the functionality of another function self.atom(token). self.atom: Numbers become numbers; every other token is a symbol.",
        trainable=True)
    def test_atom(self):
        token = ')'
        self.atom(token)

    def true_test_atom(self):
        assert repr(str(self.atom("1"))) == repr(str(1)) or (self.atom("1") == 1)
        assert repr(str(self.atom("a"))) == repr(str("a")) or (self.atom("a") == "a")
        assert repr(str(self.atom("1.2"))) == repr(str(1.2)) or (self.atom("1.2") == 1.2)

    @trace_op(description="[tokenize] Convert a string of characters into a list of tokens.",
              trainable=True)
    def tokenize(self, chars):
        "Convert a string of characters into a list of tokens."
        return chars

    @trace_op(
        description="[test_tokenize] Test the functionality of another function self.tokenize(str). self.tokenize: Convert a string of characters into a list of tokens.",
        trainable=True)
    def test_tokenize(self):
        token = ""
        self.tokenize(token)

    def true_test_tokenize(self):
        assert self.tokenize("1 + 2") == ['1', '+', '2']
        assert self.tokenize("1 + (2 * 3)") == ['1', '+', '(', '2', '*', '3', ')']

    @trace_op(description="[parse] Read a Scheme expression from a string.",
              trainable=True)
    def parse(self, program):
        self.tokenize
        self.read_from_tokens

        return program

    @trace_op(description="[test_parse] Test the functionality of another function self.parse(program). self.parse: Read a Scheme expression from a string.",
              trainable=True)
    def test_parse(self):
        program = ""
        self.parse(program)

    def true_test_parse(self):
        assert self.parse('(1 + (2 * 3))') == [1, '+', [2, '*', 3]]

    @trace_op(description="[read_from_tokens] Read an expression from a sequence of tokens.")
    def read_from_tokens(self, tokens):
        self.atom
        return tokens  # Return the fully parsed expression

    @trace_op(description="[test_read_from_tokens] Test the functionality of another function self.parse(program). self.read_from_tokens: Read an expression from a sequence of tokens.")
    def test_read_from_tokens(self):
        tokens = ['1']
        self.read_from_tokens(tokens)

    def true_test_read_from_tokens(self):
        assert self.read_from_tokens(['(', '1', '+', '(', '2', '*', '3', ')', ')']) == [1, '+', [2, '*', 3]]

global_env = Environment()
global_env.update(standard_env())

# make a list of
# this structure of code cannot be done
# good to think about it (current limitations)

def eval_expression(x, env=global_env):
    "Evaluate an expression in an environment."
    # we first unpack

    if isinstance(x, Node):
        x = x.data

    if isinstance(x, str):
        return env.data.find(x)[x]
    elif not isinstance(x, list):
        return x

    op, *args = x
    if op == 'quote':
        return args[0]
    elif op == 'define':
        (name, exp) = args
        env.data[name] = eval_expression(exp, env)
    elif op == 'lambda':
        (parms, body) = args
        return lambda *args: eval_expression(body, node(Environment(parms, args, env)))
    elif op == 'if':
        (test, conseq, alt) = args
        exp = (conseq if eval_expression(test, env) else alt)
        return eval_expression(exp, env)
    else:
        proc = eval_expression(op, env)
        vals = [eval_expression(arg, env) for arg in args]
        return proc(*vals)


test_program = [
    "(define r 10)",
    "(define circle-area (lambda (r) (* pi (* r r))))",
    "(circle-area 3)",
    "(quote (1 2 3))",
    "(if (> 10 20) (quote true) (quote false))"
]

global_env = node(global_env)

for expr in test_program:
    parsed_expr = parse(expr)
    result = eval_expression(parsed_expr, global_env)
    if isinstance(result, list):
        print([x.data for x in result])
    else:
        print(result)  # Outputs for each expression