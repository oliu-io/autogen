import autogen
from autogen.trace import trace_op, node
from autogen.trace.trace_ops import TraceExecutionError
from autogen.trace.optimizers import FunctionOptimizer, LlmAsOptimizer
from autogen.trace.nodes import GRAPH

from number_synthetic import NumericalProgramSampler
from verbal_gym.agents.basic_agent import BasicAgent
from verbal_gym.agents.llm import Autgen

# Multi-round optimization
# run 3, 10, 30 steps

import re
import os
import random
import pickle
import json

from tqdm import tqdm


def get_dataset(n=100, init_seed=1111):
    random.seed(init_seed)
    seeds = []
    for _ in range(n):
        seeds.append(random.randint(0, 99999))
    return seeds


def simple_newton_test(program, seed):
    h = 0.1
    x = 0.5
    x_0, x_1, x_2 = x - h, x, x + h
    y_0, y_1, y_2 = program(x_0, seed=seed), program(x_1, seed=seed), program(x_2, seed=seed)
    # by increasing x, we are getting smaller
    if y_2 < y_1 or y_1 < y_0:
        return True


def get_hard_dataset(n=100, init_seed=1111):
    # just do a filtering here
    random.seed(init_seed)
    seeds = []
    while len(seeds) < n:
        seed = random.randint(0, 99999)
        try:
            program = NumericalProgramSampler(chain_length=args.c, param_num=args.p, logic_prob=0, max_gen_var=args.g,
                                              seed=seed, verbose=False)
        except:
            # if we get an error program, we just move on
            continue

        # we do two checks:
        # 1. The output can't be larger than 1e5
        if program.goal_output > 1e5:
            continue
        # 2. The input/output can't move in the same direction
        pass_test = simple_newton_test(program, seed)
        if not pass_test:
            continue

        seeds.append(seed)

    return seeds


def optimize(program, program_id, optimizer, x, n_steps, verbose=False, mask=None):

    history = []
    feedback = ""
    for i in tqdm(range(n_steps)):
        GRAPH.clear()

        if verbose:
            print(f"Step: {i}")

        if feedback.lower() == "Success.".lower():
            break

        try:
            output = program(x, seed=program_id)
            feedback = program.feedback(output.data)
        except TraceExecutionError as e:
            output = e.exception_node
            feedback = output.data

        history.append((x.data, output.data, program.goal_input, program.goal_output, feedback))  # logging

        optimizer.zero_feedback()
        optimizer.backward(output, feedback)
        if verbose:
            print(f"variable={x.data}, output={output.data}, feedback={feedback}")  # logging
        optimizer.step(mask=mask)

    history.append((x.data, output.data, program.goal_input, program.goal_output, feedback))  # logging
    return history


def run_exp(masking=False, optimizer_name=None):
    problem_ids = get_hard_dataset(n=args.n)
    n_steps = args.steps  # we allow 10 optimization steps

    traj_for_all_problems = []
    for i in tqdm(range(len(problem_ids))):
        # multi-param might be interesting but I don't know how to adapt this pipeline for it
        program = NumericalProgramSampler(chain_length=args.c, param_num=args.p, logic_prob=0, max_gen_var=args.g,
                                          seed=problem_ids[i])
        x = node(-1.0, "input_x", trainable=True)
        if optimizer_name == 'opro':
            optimizer = LlmAsOptimizer([x], config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))
        else:
            optimizer = FunctionOptimizer([x], config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))

        mask = ['#Documentation', '#Code', '#Inputs', '#Others'] if masking else None
        try:
            history = optimize(program, problem_ids[i], optimizer, x, n_steps, verbose=args.verbose, mask=mask)
        except:
            print("skipping this problem due to an optimization error")
            continue
        traj_for_all_problems.append(history)

    os.makedirs("results", exist_ok=True)
    agent_name = 'trace_agent' if not masking else 'masked_trace_agent'
    if optimizer_name:
        agent_name = agent_name + f'_{optimizer_name}'

    with open(f"results/{agent_name}_number_synth_traj_{args.n}_c_{args.c}_g_{args.g}_p_{args.p}.pkl", "wb") as f:
        pickle.dump(traj_for_all_problems, f)

def rollout(program, program_id, agent, x, n_steps, verbose=False):
    # similar to "optimize" but we just rollout for all agents
    history = []
    feedback = ""
    for i in tqdm(range(n_steps)):
        if verbose:
            print(f"Step: {i}")

        if feedback.lower() == "Success.".lower():
            break

        try:
            output = program(x, seed=program_id)
            feedback = program.feedback(output)
        except Exception as e:
            output = "NaN"
            feedback = str(e)

        # issue: I think the timestep here is wrong
        observation = f"{output}"

        try:
            # to handle context length issue
            action = agent.act(observation, feedback)
        except Exception as e:
            pass

        try:
            pattern = r'\d+(?:\.\d+)?'
            match = re.search(pattern, action.strip())
            first_number = match.group()
            x = float(first_number)
        except:
            # we keep the original x if there's an error
            print(f"error in parsing:\n {action.strip()}\n")

        if verbose:
            print(f"variable={x}, output={output}, feedback={feedback}")  # logging

        history.append((x, output, program.goal_input, program.goal_output, feedback))

    history.append((x, output, program.goal_input, program.goal_output, feedback))

    return history


def run_basic_agent_exp(agent_type='basic'):
    llm = Autgen()
    if agent_type == 'basic':
        agent = BasicAgent(llm)
    else:
        raise Exception("Agent type not implemented")

    # problem_ids = get_dataset(n=args.n)
    problem_ids = get_hard_dataset(n=args.n)
    n_steps = args.steps  # we allow 10 optimization steps

    instruction = (
        "You are choosing an input that after some operations will result in an output. You will observe some feedback telling you whether"
        "your output is too large or too small to hit a hidden goal value. You need to choose your input in order to hit that goal output value.")

    agent.reset(docstring=instruction)

    traj_for_all_problems = []
    for i in tqdm(range(len(problem_ids))):
        # multi-param might be interesting but I don't know how to adapt this pipeline for it
        program = NumericalProgramSampler(chain_length=args.c, param_num=args.p, logic_prob=0, max_gen_var=args.g,
                                          seed=problem_ids[i])
        x = -1.0

        history = rollout(program, problem_ids[i], agent, x, n_steps, verbose=args.verbose)
        traj_for_all_problems.append(history)

        agent.reset(instruction)

    os.makedirs("results", exist_ok=True)
    with open(f"results/{agent_type}_agent_number_synth_traj_{args.n}_c_{args.c}_g_{args.g}_p_{args.p}.pkl", "wb") as f:
        pickle.dump(traj_for_all_problems, f)

def run_torch_exp():
    import torch

    x = torch.tensor([5.], requires_grad=True)

    problem_ids = get_hard_dataset(n=args.n)
    n_steps = args.steps  # we allow 10 optimization steps

    traj_for_all_problems = []
    for i in tqdm(range(len(problem_ids))):
        # multi-param might be interesting but I don't know how to adapt this pipeline for it
        program = NumericalProgramSampler(chain_length=args.c, param_num=args.p, logic_prob=0, max_gen_var=args.g,
                                          seed=problem_ids[i])
        x = torch.tensor([-1.0], requires_grad=True)

        # do greadient descent on x, 10 steps
        history = []
        optimizer = torch.optim.Adam([x], lr=1e-1)
        for _ in range(n_steps):
            optimizer.zero_grad()
            y = program(x, seed=problem_ids[i])
            loss = torch.abs(y - program.goal_output)
            loss.backward()
            optimizer.step()
            history.append((float(x.detach()), float(y.detach()), program.goal_input, program.goal_output, "Gradient feedback"))

        history.append(
            (float(x.detach()), float(y.detach()), program.goal_input, program.goal_output, "Gradient feedback"))

        traj_for_all_problems.append(history)

    os.makedirs("results", exist_ok=True)
    agent_name = 'torch'
    agent_name = agent_name + f'_sgd'

    with open(f"results/{agent_name}_number_synth_traj_{args.n}_c_{args.c}_g_{args.g}_p_{args.p}.pkl", "wb") as f:
        pickle.dump(traj_for_all_problems, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=100)
    parser.add_argument('--c', type=int, default=7)
    parser.add_argument('--g', type=int, default=5)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--setup', type=str, default='trace', help='trace, opro, masked, torch, agent')
    parser.add_argument('--agent_type', type=str, default='basic', help='basic...')

    args = parser.parse_args()

    if args.setup == 'agent':
        run_basic_agent_exp(args.agent_type)
    elif args.setup == 'opro':
        run_exp(optimizer_name='opro')
    elif args.setup == 'masked':
        run_exp(masking=True)
    elif args.setup == 'trace':
        run_exp()
    elif args.setup == 'torch':
        run_torch_exp()
