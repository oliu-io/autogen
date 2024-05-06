from autogen.trace import node, trace_op
from autogen.trace.modules import apply_op
from autogen.trace.modules import NodeContainer
import autogen.trace.operators as ops

# ========== Case 1 ==========

"""
Not able to tracing through func_a (updating func_a's parameter)
"""
@trace_op(description="[func_a] Returns a+1", trainable=True)
def func_a(a):
    return a + 1

@trace_op(description="[func_b] Returns b+1", trainable=True)
def func_b(b):
    return func_a(b) + 1

def test_nested_function_visibility():
    x = node(3)
    y = func_b(x)
    y.backward(visualize=True)

# ========== Case 2 ==========

"""
Updating external variables
"""