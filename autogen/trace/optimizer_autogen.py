import gym
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.trace.optimizers import Optimizer, PropagateStrategy
from autogen.trace.trace import trace

"""
This file includes training utility functions specifically 
made for AutoGen
"""


# TODO: add the "filtering" rule
# TODO: add the expand and select strategy
# TODO: might need to unify the two, or refactor these
def train_with_wrapped_env(env_agent: UserProxyAgent, agent: ConversableAgent, optimizer: Optimizer,
                           steps: int, propagate_fn=PropagateStrategy.retain_last_only_propagate, verbose: bool = False):
    # we assume the environment is wrapped around a user agent
    # right now the function is only written for bandit environments
    # Note: the training happens in-place

    info = {}
    info['prompt_traj'] = []  # a list of dictionary {'content': prompt, 'role': 'system'}
    info['rewards'] = []  # a list of rewards

    early_break = False
    # optimization steps
    for k in range(steps):
        info['prompt_traj'].append(optimizer.parameters[0].data)

        if verbose:
            print(f"Prompt at step {k}:", optimizer.parameters[0].data)

        init_obs = env_agent.get_starting_message()
        env_agent.initiate_chat(agent, message=init_obs, clear_history=True, silent=not verbose)
        feedback = env_agent.last_message_node().data['content']

        info['rewards'].append(env_agent.reward_history[-1])

        if env_agent.reward_history[-1] == 1.0:
            if verbose:
                print("Reached highest reward.")

            early_break = True
            break

        opt_step_with_feedback(feedback, env_agent, agent, optimizer, propagate_fn, verbose)

    if not early_break:
        # we add the last updated prompt into the history
        info['prompt_traj'].append(optimizer.parameters[0].data)
        # otherwise we don't (because the previous prompt is already the best)

    return info

def train_with_env(env: gym.Env, agent: ConversableAgent, optimizer: Optimizer,
                  steps: int, feedback_verbalize: callable, propagate_fn=PropagateStrategy.retain_last_only_propagate,
                  verbose: bool = False):
    # we assume the environment is wrapped around a user agent
    # right now the function is only written for bandit environments
    # Note: the training happens in-place

    # feedback_verbalize: takes in `observation`, `feedback`, `reward` and returns a string

    # we provide a fake user agent
    # and query feedback from the environment
    user_agent = trace(UserProxyAgent)(name="user agent", human_input_mode="NEVER",
                                       default_auto_reply="TERMINATE")

    opt_info = {}
    opt_info['prompt_traj'] = []  # a list of dictionary {'content': prompt, 'role': 'system'}
    opt_info['rewards'] = []  # a list of rewards

    early_break = False
    # optimization steps
    for k in range(steps):
        opt_info['prompt_traj'].append(optimizer.parameters[0].data)

        if verbose:
            print(f"Prompt at step {k}:", optimizer.parameters[0].data)

        obs, info = env.reset()
        init_obs = obs['instruction']

        user_agent.initiate_chat(agent, message=init_obs, clear_history=True, silent=not verbose)

        last_message = agent.last_message_node(user_agent, role='assistant')

        next_obs, reward, terminated, truncated, info = env.step(last_message['content'])
        success = info['success']
        feedback = feedback_verbalize(next_obs['observation'], next_obs['feedback'], reward)

        opt_info['rewards'].append(reward)

        if reward == 1.0:
            if verbose:
                print("Reached highest reward.")

            early_break = True
            break

        opt_step_with_feedback(feedback, user_agent, agent, optimizer, propagate_fn, verbose)

    if not early_break:
        # we add the last updated prompt into the history
        opt_info['prompt_traj'].append(optimizer.parameters[0].data)
        # otherwise we don't (because the previous prompt is already the best)

    return opt_info

def opt_step_with_feedback(feedback: str, env_agent: UserProxyAgent, agent: ConversableAgent, optimizer: Optimizer,
                           propagate_fn=PropagateStrategy.retain_last_only_propagate, verbose: bool = False):
    last_message = agent.last_message_node(env_agent, role='assistant')
    optimizer.zero_feedback()
    last_message.backward(feedback, propagate_fn, retain_graph=False)
    optimizer.step()
