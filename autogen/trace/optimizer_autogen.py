from typing import Tuple

import gym
import numpy as np
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.trace.optimizers import Optimizer
from autogen.trace.propagators import retain_last_only_propagate
from autogen.trace.trace import trace
from textwrap import dedent, indent

"""
This file includes training utility functions specifically
made for AutoGen
"""


# TODO: add the "filtering" rule
# TODO: add the expand and select strategy
# TODO: might need to unify the two, or refactor these
def train_with_wrapped_env(
    env_agent: UserProxyAgent,
    agent: ConversableAgent,
    optimizer: Optimizer,
    steps: int,
    propagate_fn=None,
    verbose: bool = False,
):
    # we assume the environment is wrapped around a user agent
    # right now the function is only written for bandit environments
    # Note: the training happens in-place
    if propagate_fn is None:
        propagate_fn = retain_last_only_propagate()

    info = {}
    info["prompt_traj"] = []  # a list of dictionary {'content': prompt, 'role': 'system'}
    info["rewards"] = []  # a list of rewards

    early_break = False
    # optimization steps
    for k in range(steps):
        info["prompt_traj"].append(optimizer.parameters[0].data)

        if verbose:
            print(f"Prompt at step {k}:", optimizer.parameters[0].data)

        init_obs = env_agent.get_starting_message()
        agent.clear_history()  # this is quite important; also can't wait till zero_grad() step...

        env_agent.initiate_chat(agent, message=init_obs, clear_history=True, silent=not verbose)
        feedback = env_agent.last_message_node().data["content"]

        info["rewards"].append(env_agent.reward_history[-1])

        if env_agent.reward_history[-1] == 1.0:
            if verbose:
                print("Reached highest reward.")

            early_break = True
            break

        last_message = agent.last_message_node(env_agent, role="assistant")
        opt_step_with_feedback(feedback, last_message, optimizer, propagate_fn, verbose)

    if not early_break:
        # we add the last updated prompt into the history
        info["prompt_traj"].append(optimizer.parameters[0].data)
        # otherwise we don't (because the previous prompt is already the best)

    return info


def train_with_env(
    env: gym.Env,
    agent: ConversableAgent,
    optimizer: Optimizer,
    steps: int,
    feedback_verbalize: callable,
    propagate_fn=retain_last_only_propagate(),
    verbose: bool = False,
    max_reward=None,
):
    # we assume the environment is wrapped around a user agent
    # right now the function is only written for bandit environments
    # Note: the training happens in-place

    # feedback_verbalize: takes in `observation`, `feedback`, `reward` and returns a string
    # max_reward: terminates when it receives this reward

    # we provide a fake user agent
    # and query feedback from the environment
    user_agent = trace(UserProxyAgent)(
        name="user agent", human_input_mode="NEVER", default_auto_reply="TERMINATE", code_execution_config=False
    )

    opt_info = {}
    opt_info["prompt_traj"] = []  # a list of dictionary {'content': prompt, 'role': 'system'}
    opt_info["rewards"] = []  # a list of rewards

    early_break = False
    # optimization steps
    for k in range(steps):
        opt_info["prompt_traj"].append(optimizer.parameters[0].data)

        if verbose:
            print(f"Prompt at step {k}:", optimizer.parameters[0].data)

        obs, info = env.reset()
        init_obs = obs["instruction"]

        agent.clear_history()  # this is quite important; also can't wait till zero_grad() step...

        user_agent.initiate_chat(agent, message=init_obs, clear_history=True, silent=not verbose)

        last_message = agent.last_message_node(user_agent, role="assistant")

        next_obs, reward, terminated, truncated, info = env.step(last_message.data["content"])
        info["success"]
        feedback = feedback_verbalize(next_obs["observation"], next_obs["feedback"], reward)

        opt_info["rewards"].append(reward)

        if reward == max_reward:
            if verbose:
                print("Reached highest reward.")

            early_break = True
            break

        opt_step_with_feedback(feedback, last_message, optimizer, propagate_fn, verbose)

    if not early_break:
        # we add the last updated prompt into the history
        opt_info["prompt_traj"].append(optimizer.parameters[0].data)
        # otherwise we don't (because the previous prompt is already the best)

    return opt_info


def opt_step_with_feedback(
    feedback: str, last_message, optimizer: Optimizer, propagate_fn=retain_last_only_propagate(), verbose: bool = False
):
    optimizer.zero_feedback()
    last_message.backward(feedback, propagate_fn, retain_graph=True)
    optimizer.step()


class DatasetProcessor:
    """
    Override this class to provide custom functions for giving feedback
    """

    def __init__(self, input_field="", answer_field="", reward_fn=None):
        self.input_field = input_field
        self.answer_field = answer_field
        self.reward_fn = reward_fn

    def get_obs(self, row):
        return row[self.input_field]

    def get_answer(self, row):
        return row[self.answer_field]

    def generate_feedback(self, predicted_answer, reference_answer) -> Tuple[str, float]:
        score = 0
        if self.reward_fn is not None:
            score = self.reward_fn(predicted_answer, reference_answer)

        feedback = dedent(
            f"""
        Your answer is {predicted_answer}

        The correct answer is:
        {reference_answer}
        """
        )

        if self.reward_fn is not None:
            message = f"""Score: {score}\n\n"""
        else:
            message = ""

        message += f"Feedback: {feedback}\n\n"

        return message, score


# Currently, there is a caching issue -- output of LLMs don't change
# Needs fixing
def train_with_datasets(
    dataset,
    dataset_processor: DatasetProcessor,
    generate_answer: callable,
    optimizer: Optimizer,
    steps: int,
    propagate_fn=retain_last_only_propagate(),
    verbose: bool = False,
    seed=None,
):
    # generate_answer: context/input -> node
    # we assume a huggingface API

    if seed is not None:
        np.random.seed(seed)

    num_examples = len(dataset)
    if steps < num_examples:
        training_indices = np.random.choice(num_examples, steps, replace=False)
    else:
        full_iter_count = steps // num_examples
        remainder = steps % num_examples
        training_indices = np.random.choice(num_examples, remainder, replace=False)
        training_indices = np.concatenate([np.arange(num_examples)] * full_iter_count + [training_indices])

    opt_info = {}
    opt_info["prompt_traj"] = []  # a list of dictionary {'content': prompt, 'role': 'system'}
    opt_info["rewards"] = []  # a list of rewards

    # optimization steps
    for k in range(steps):
        opt_info["prompt_traj"].append(optimizer.parameters[0].data)

        if verbose:
            print(f"Prompt at step {k}:", optimizer.parameters[0].data)

        row = dataset[int(training_indices[k])]
        init_obs = dataset_processor.get_obs(row)

        # user_agent.initiate_chat(agent, message=init_obs, clear_history=True, silent=not verbose)
        # last_message = agent.last_message_node(user_agent, role='assistant')
        last_message = generate_answer(init_obs)

        reference_answer = dataset_processor.get_answer(row)
        predicted_answer = last_message.data["content"]

        feedback, reward = dataset_processor.generate_feedback(predicted_answer, reference_answer)

        opt_info["rewards"].append(reward)

        opt_step_with_feedback(feedback, last_message, optimizer, propagate_fn, verbose)

    return opt_info
