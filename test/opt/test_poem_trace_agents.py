"""
We test if trace works on different agent designs
"""

import numpy as np
import random
from tqdm import tqdm

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, Agent
from poem_agents import PoemAgent, PoemStudentAgent, PoemExtractor, config_list
from env_wrapper import LLFBenchUserAgent

from autogen.trace.trace import trace
from autogen.trace.propagators.propagators import retain_last_only_propagate


def test_single_agent():
    student_agent = trace(PoemStudentAgent)(seed=13)
    user_agent = trace(LLFBenchUserAgent)(
        env_name="llf-poem-Haiku-v0", llm_config={"temperature": 0.0, "config_list": config_list}
    )

    init_obs = user_agent.get_starting_message()
    user_agent.initiate_chat(student_agent, message=init_obs)

    feedback = user_agent.last_message_node().data["content"]
    last_message = student_agent.last_message_node()

    last_message.backward(feedback, retain_last_only_propagate(), retain_graph=False, visualize=False)

    assert (
        list(student_agent.parameters[0].feedback.values())[0][0] == feedback
    ), "The feedback should backprop all the way to the parameter nodes"
    print("TEST: feedback backpropagation successful!")


def test_nested_agent():
    poem_agent = PoemAgent(seed=13)

    user_agent = trace(LLFBenchUserAgent)(
        env_name="llf-poem-Haiku-v0", llm_config={"temperature": 0.0, "config_list": config_list}
    )

    init_obs = user_agent.get_starting_message()
    user_agent.initiate_chat(poem_agent, message=init_obs, clear_history=True)

    last_message = poem_agent.chat_message_nodes[user_agent][-2]
    feedback = user_agent.last_message_node().data["content"]
    last_message.backward(feedback, retain_last_only_propagate(), retain_graph=False, visualize=False)

    assert (
        list(poem_agent.student_agent.parameters[0].feedback.values())[0][0] == feedback
    ), "The feedback should backprop all the way to the parameter nodes"
    print("TEST: feedback backpropagation successful!")


def test_agent_designs(agent_design):
    if agent_design == "single":
        test_single_agent()
    elif agent_design == "nested":
        test_nested_agent()
    elif agent_design == "all":
        test_single_agent()
        test_nested_agent()
    else:
        raise ValueError(f"Agent design {agent_design} not supported yet")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("agent_design", type=str, default="single", help="single|nested|all")
    # 'groupchat', 'sequential'
    test_agent_designs(**vars(parser.parse_args()))
