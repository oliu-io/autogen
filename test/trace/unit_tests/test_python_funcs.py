"""
This tests all the functions of Lisp Interpreter
To make sure things work
"""
from autogen.trace.nodes import node, Node
from autogen.trace import operators as ops
from autogen.trace.trace_ops import trace_op
import math

@trace_op(
    description="[get_env] Return a new env inside env with parms mapped to their corresponding args, and env as the new env's outer env.",
    trainable=True)
def get_env(parms, args, env=None):
    new_env = {'_outer': env}
    for (parm, arg) in zip(parms, args):
        new_env[parm] = arg
    return new_env


def test_get_env():
    env = get_env(node(["a", "b"]), node([1, 2]))
    assert isinstance(env, Node)
    assert env.data == {'_outer': None, 'a': 1, 'b': 2}


# import must be local, otherwise we can't lazy execute
@trace_op(description="[get_math] Get a dictionary mapping math library function names to their functions.", trainable=True)
def get_math():
    d = {}
    for name in dir(math):
        if name[:2] != '__':
            d[name] = getattr(math, name)
    return d


def test_empty_inputs():
    result = get_math()
    assert isinstance(result, Node)
    result.backward()

    result = get_ops()
    assert isinstance(result, Node)
    result.backward()


@trace_op(description="[get_ops] Get a dictionary mapping math library function names to their functions.", trainable=True)
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

@trace_op(description="[apply_fn_dict_key] Return the value of fn_dict_generator()[key](*args_list) in standard_env.",
          unpack_input=False, trainable=True, catch_execution_error=False, node_dict=None)
def apply_fn_dict_key(fn_dict_generator, key, args_list):
    fn_dict = fn_dict_generator()
    return fn_dict[key](*args_list)

def test_apply_fn_dict_key():
    fn_dict_generator = get_ops
    key = "+"
    args_list = node([1, 2])
    result = apply_fn_dict_key(fn_dict_generator, key, args_list)
    assert result.data == 3

    result.backward(visualize=True)

test_get_env()
test_empty_inputs()
test_apply_fn_dict_key()

@trace_op(
    description="[get_simple_math] Get a dictionary mapping 'abs', 'min', 'max', 'not', 'round' to their functions.",
    trainable=True)
def get_simple_math():
    return {'abs': abs, 'min': min, 'max': max, 'not': lambda x: not x, 'round': round}

@trace_op(
    description="[standard_env] An environment with some Scheme standard procedures. Start with an environment and update it with standard functions.",
    node_dict=None, trainable=True, catch_execution_error=False)
def standard_env(includes=['math', 'ops', 'simple_math']):
    env = {'_outer': None}
    if 'math' in includes:
        env.update(get_math())
    if 'ops' in includes:
        env.update(get_ops())
    if 'simple_math' in includes:
        env.update(get_simple_math())
    return env

def test_standard_env():
    env = standard_env()
    assert isinstance(env, Node)

# this throws an error
test_standard_env()