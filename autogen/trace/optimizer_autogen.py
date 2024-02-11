from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.trace.optimizers import Optimizer, PropagateStrategy

"""
This file includes training utility functions specifically 
made for AutoGen
"""

# TODO: add the "filtering" rule
# TODO: add the expand and select strategy
def train(env_agent: UserProxyAgent, poem_agent: ConversableAgent, optimizer: Optimizer,
          steps: int, propagate_fn=PropagateStrategy.retain_last_only_propagate, verbose: bool = False):
    # we assume the environment is wrapped around a user agent
    # right now the function is only written for bandit environments
    # Note: the training happens in-place

    info = {}
    info['prompt_traj'] = []  # a list of dictionary {'content': prompt, 'role': 'system'}
    info['rewards'] = []  # a list of rewards

    early_break = False
    for k in range(steps):
        info['prompt_traj'].append(poem_agent.parameters[0].data)

        if verbose:
            print(f"Prompt at step {k}:", poem_agent.parameters[0].data)

        init_obs = env_agent.get_starting_message()
        env_agent.initiate_chat(poem_agent, message=init_obs, clear_history=True, silent=not verbose)
        feedback = env_agent.last_message_node().data['content']

        info['rewards'].append(env_agent.reward_history[-1])

        if env_agent.reward_history[-1] == 1.0:
            if verbose:
                print("Reached highest reward.")

            early_break = True
            break

        last_message = poem_agent.last_message_node(env_agent, role='assistant')

        optimizer.zero_feedback()
        last_message.backward(feedback, propagate_fn, retain_graph=False)
        optimizer.step()

    if not early_break:
        # we add the last updated prompt into the history
        info['prompt_traj'].append(poem_agent.parameters[0].data)
        # otherwise we don't (because the previous prompt is already the best)

    return info