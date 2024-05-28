from autogen.trace import trace_op, node, Module
from autogen.trace.nodes import ExceptionNode
from autogen.trace.trace_ops import TraceExecutionError, trace_class
from autogen.trace.optimizers import FunctionOptimizerV2, FunctionOptimizerV2Memory, OPRO
from autogen.trace.nodes import GRAPH
from battleship import BattleshipBoard
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import pickle
import os
from datetime import datetime


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


def rollout(policy, board_width, board_height, num_each_type, exclude_ships, horizon):
    board = BattleshipBoard(board_width, board_height, num_each_type=num_each_type, exclude_ships=exclude_ships)
    rewards = []
    obs = board.get_shots()  # init observation
    for i in range(horizon):
        fire_again, terminal = True, False
        turn_reward = 0
        while fire_again:
            output = policy(obs)
            obs, reward, terminal, feedback = user_fb_for_placing_shot(board, output)  # not traced
            hit = reward > 0  # if hit, we are allowed to go again
            fire_again = hit and not terminal
            turn_reward += reward

        rewards.append(turn_reward)
        if terminal:
            break

    rewards = np.array(rewards)
    return rewards


def eval_policy(policy, args):
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    scores = []
    for _ in range(args.n_eval_episodes):
        # board_width = np.random.randint(args.board_size, int(2 * args.board_size))
        # board_height = np.random.randint(args.board_size, int(2 * args.board_size))
        board_width = args.board_size
        board_height = args.board_size
        # horizon = (board_width * board_height) // 2  # we don't have that many chances (imagine an opponent)
        horizon = 25 # int(17 * 1.5)
        policy.init(board_width, board_height)
        rewards = rollout(policy, board_width, board_height, args.num_each_type, args.exclude_ships, horizon)
        scores.append(rewards.sum() / horizon)
    scores = np.array(scores)
    print(f"Scores: {scores.mean()} ({scores.std()})")
    return scores


def run(args):
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # init variable
    # Given a map, select a valid coordinate to earn reward.

    # a wrapper for the eval api

    class Policy(Module):
        def init(self, width, height):
            pass

        def __call__(self, map):
            return self.select_coordinate(map).data

        @trace_op(trainable=True)
        def select_coordinate(self, map):
            """
            Given a map, select a target coordinate in a game. In map, O denotes misses, X denotes successes, and . denotes unknown positions.
            """
            return [0, 0]

    @trace_class
    class Policy2:
        def init(self, width, height):
            pass

        def __call__(self, map):
            return self.select_coordinate(map).data

        def select_coordinate(self, map):
            plan = self.reason(map)

            act = self.act
            output = act(map, plan)
            return output

        @trace_op(trainable=True)
        def act(self, map, plan):
            """
            Given a map, select a target coordinate in a game. On map, O denotes misses, X denotes successes, and . denotes unknown positions.
            """
            return

        @trace_op(trainable=True)
        def reason(self, map) -> str:
            """
            Given a map, analyze the board in a game. On map, O denotes misses, X denotes successes, and . denotes unknown positions.
            """
            return [0, 0]

    # if args.optimizer == "FunctionOptimizerV2Memory":
    #     policy = Policy2()
    # else:
    #     policy = Policy()
    policy = Policy2()

    def reset_board():
        # square board
        board = BattleshipBoard(
            args.board_size, args.board_size, num_each_type=args.num_each_type, exclude_ships=args.exclude_ships
        )
        if args.visualize:
            print("Ground State Board")
            board.visualize_board()
        return board

    optimizer = args.opt_cls(policy.parameters())

    # writer = SummaryWriter(args.logdir)
    log = defaultdict(list)

    # init eval
    returns = eval_policy(policy, args)
    log["returns"].append(returns)
    # writer.add_scalar("Evaluation/mean score", returns.mean(), 0)

    feedback = ""
    rewards = []
    # This is an online optimization problem. we have the opportunity to
    # keep changing the function with each round of interaction
    board = reset_board()

    i = 0
    while i < args.max_calls:
        GRAPH.clear()
        obs = node(board.get_shots())  # init observation

        try:
            output = policy.select_coordinate(obs)
            obs, reward, terminal, feedback = user_fb_for_placing_shot(board, output.data)  # not traced
        except TraceExecutionError as e:  # this is a retry
            output = e.exception_node
            feedback = output.data
            reward, terminal = 0, False

        if terminal:
            board = reset_board()
            obs = node(board.get_shots())  # init observation

        # Update
        optimizer.zero_feedback()
        optimizer.backward(output, feedback)
        optimizer.step(verbose=False)

        # Logging
        if not isinstance(output, ExceptionNode):
            rewards.append(reward)
            if args.visualize:
                print("Obs:")
                board.visualize_shots()
                print(f"output={output.data}, feedback={feedback}, variables=\n")  # logging
                for p in optimizer.parameters:
                    print(p.name, p.data)
            try:
                returns = eval_policy(policy, args)
                log["returns"].append(returns)
                log["instant reward"].append(reward)
                parameter_dict = {name: n.data for name, n in policy.parameters_dict().items()}
                log['parameters'].append(parameter_dict)
                # writer.add_scalar("Evaluation/mean score", returns.mean(), i + 1)  # i+1 to account for the initial log
                # writer.add_scalar("Training/instant reward", reward, i)
                #
                # writer.flush()
            except Exception:
                pass
            i += 1

        pickle.dump(log, open(f"{args.logdir}/log.pkl", "wb"))

        # 0.420 should be the max for 8x8 board
        if log['returns'][-1].mean() > 0.41:
            break

    rewards = np.array(rewards)
    print(f"Cumulative rewards: {rewards.sum()}")
    print(f"Average rewards: {rewards.mean()}")
    return rewards


class BasicEnumerator:
    def init(self, width, height):
        self.width = width
        self.height = height
        self.i = 0
        self.j = 0

    def __call__(self, obs):
        if self.i == self.width:
            self.i = 0
            self.j += 1
        if self.j == self.height:
            raise StopIteration
        self.i += 1
        return [self.i, self.j]


class RandomPolicy(BasicEnumerator):
    def __call__(self, obs):
        return [np.random.randint(self.width), np.random.randint(self.height)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, default=8)
    parser.add_argument("--num_each_type", type=int, default=1)
    parser.add_argument("--exclude_ships", nargs="+", type=str, default=())  # "C", "B"
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_calls", type=int, default=20)
    parser.add_argument("--memory_size", type=int, default=0)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="battleship_results")
    parser.add_argument("--optimizer", type=str, default="FunctionOptimizerV2Memory")
    args = parser.parse_args()

    # Log
    args.logdir += (
        f"/{args.optimizer}_mem{args.memory_size}/seed_{args.seed}_size{args.board_size}_num{args.num_each_type}"
    )
    args.logdir += datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.logdir, exist_ok=True)

    pickle.dump(args, open(f"{args.logdir}/args.pkl", "wb"))

    print("Enumeration")
    scores = eval_policy(BasicEnumerator(), args)
    pickle.dump(scores, open(f"{args.logdir}/enumeration_scores.pkl", "wb"))
    print("Random")
    scores = eval_policy(RandomPolicy(), args)
    pickle.dump(scores, open(f"{args.logdir}/random_policy_scores.pkl", "wb"))
    print("Optimizing policy")

    if args.optimizer == "FunctionOptimizerV2Memory":
        args.opt_cls = lambda *_args, **_kwargs: FunctionOptimizerV2Memory(
            *_args, **_kwargs, memory_size=args.memory_size
        )
    elif args.optimizer == "OPRO":
        args.opt_cls = lambda *_args, **_kwargs: OPRO(
            *_args, **_kwargs, memory_size=args.memory_size
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    run(args)
