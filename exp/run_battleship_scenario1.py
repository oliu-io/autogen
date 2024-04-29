import autogen
from autogen.trace import trace_op, node
from autogen.trace.trace_ops import TraceExecutionError
from autogen.trace.optimizers import FunctionOptimizer
from autogen.trace.nodes import GRAPH

from battleship import BattleshipBoard

# ===== Scenario 1 ===== #
@trace_op("[select_coordinate] Given a map, select a valid coordinate to see if we can earn reward.", trainable=True)
def select_coordinate(map):
    """
    Given a map, select a valid coordinate. We might earn reward from this coordinate.
    """
    return [0, 0]

def user_fb_for_placing_shot(board, coords):
    # this is already a multi-step cumulative reward problem
    # obs, reward, terminal, feedback
    try:
        reward = board.check_shot(coords[0], coords[1])
        new_map = board.get_shots()
        terminal = board.check_terminate()
        return new_map, reward, terminal, f"Got {int(reward)} reward."
    except Exception as e:
        return board.get_shots(), 0, False, str(e)


GRAPH.clear()

board = BattleshipBoard(5, 5, num_each_type=1, exclude_ships=['C', 'B'])
print("Ground State Board")
board.visualize_board()

obs = node(board.get_shots(), trainable=False)
optimizer = FunctionOptimizer([select_coordinate.parameter],
                              config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))

feedback = ""
terminal = False
cum_reward = 0
max_calls = 10

while not terminal and max_calls > 0:
    # This is also online optimization
    # we have the opportunity to keep changing the function with each round of interaction
    try:
        output = select_coordinate(obs)
        obs, reward, terminal, feedback = user_fb_for_placing_shot(board, output.data)
    except TraceExecutionError as e:
        # this is essentially a retry
        output = e.exception_node
        feedback = output.data
        terminal = False
        reward = 0

    print("Obs:")
    board.visualize_shots()

    cum_reward += reward

    optimizer.zero_feedback()
    optimizer.backward(output, feedback)

    print(f"output={output.data}, feedback={feedback}, variables=\n")  # logging
    for p in optimizer.parameters:
        print(p.name, p.data)
    optimizer.step(verbose=False)
    max_calls -= 1