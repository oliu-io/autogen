import autogen
from autogen.trace import trace_op, node
from autogen.trace.trace_ops import TraceExecutionError
from autogen.trace.optimizers import FunctionOptimizer
from autogen.trace.nodes import GRAPH

from battleship import BattleshipBoard

# ===== Scenario 0 ===== #
@trace_op("[select_coordinate] Given a map, select a valid coordinate.", trainable=True)
def select_coordinate(map):
    """
    Given a map, select a valid coordinate.
    """
    return map

def user_fb_for_coords_validity(board, coords):
    try:
        board.check_shot(coords[0], coords[1])
    except Exception as e:
        return str(e), 0


GRAPH.clear()

board = BattleshipBoard(10, 10)

x = node(board.board, trainable=False)
optimizer = FunctionOptimizer([select_coordinate.parameter],
                              config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))

feedback = ""
reward = 0

while reward != 1:
    try:
        output = select_coordinate(x)
        feedback, reward = user_fb_for_coords_validity(board, output.data)
    except TraceExecutionError as e:
        output = e.exception_node
        feedback = output.data

    optimizer.zero_feedback()
    optimizer.backward(output, feedback)

    print(f"output={output.data}, feedback={feedback}, variables=\n")  # logging
    for p in optimizer.parameters:
        print(p.name, p.data)
    optimizer.step(verbose=False)