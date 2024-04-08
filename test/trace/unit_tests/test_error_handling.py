from autogen.trace.trace_ops import trace_op
from autogen.trace.nodes import Node, node
from autogen.trace.utils import for_all_methods

x = Node(1, name="node_x")
y = Node(0, name="node_y")

def bug_program(x: Node, y: Node):
    z = x / y
    return z

try:
    bug_program(x, y)
except Exception as e:
    print(e)