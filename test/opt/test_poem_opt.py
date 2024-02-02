"""
In this file, we should have:
2. Add FeedbackEnhance
3. An optimizer that optimizes...with an agent call (20 min)
"""

import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.trace.trace import trace, node, trace_node_usage, Node
from textwrap import dedent, indent
from env_wrapper import LLFBenchUserAgent
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

from autogen.trace.optimizers import DummyOptimizer, LLMOptimizer, PropagateStrategy

from autogen import OpenAIWrapper
from autogen import Completion, ChatCompletion

def extract_text_or_completion_object(response, tool_enabled=False):
    """Extract the text or ChatCompletion objects from a completion or chat response.

    Args:
        response (ChatCompletion | Completion): The response from openai.

    Returns:
        A list of text, or a list of ChatCompletion objects if function_call/tool_calls are present.
    """
    choices = response.choices
    if isinstance(response, Completion):
        return [choice.text for choice in choices]

    TOOL_ENABLED = tool_enabled

    if TOOL_ENABLED:
        return [
            choice.message
            if choice.message.function_call is not None or choice.message.tool_calls is not None
            else choice.message.content
            for choice in choices
        ]
    else:
        return [
            choice.message if choice.message.function_call is not None else choice.message.content
            for choice in choices
        ]


# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict={
             "model": ["gpt-3.5-turbo-0613", "gpt-3.5-turbo"],
         })
assert len(config_list) > 0 

termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

sys_msg = dedent("You are a student and your teacher gives you an assignment to write a poem.")

class StudentAgent(AssistantAgent):
    def __init__(self, seed=1234):
        super().__init__(
            name="StudentAgent",
            system_message=sys_msg,
            llm_config={"temperature": 0.0, "config_list": config_list, 'cache_seed': seed},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
        )
        self.extraction_sys_prompt = dedent("""
        You are a helpful assistant.
        You extract only lines of poems in the message from the student, ignore any part of the message that is not related to the poem.
        You should only reply with the poem string extracted from the user's input.
        """)
        self.register_reply(autogen.ConversableAgent, StudentAgent._generate_reply_for_user, position=5)

    def _generate_reply_for_user(self,
                         messages: Optional[List[Dict]] = None,
                         sender: Optional[autogen.Agent] = None,
                         config: Optional[Any] = None,
                         ):
        if messages is None:
            return True, None
        # if there is a response, we try to extract it
        response = self.client.create(messages=[{
            "role": "system",
            "content": self.extraction_sys_prompt,
        }, {"role": "user", "content": messages[-1]['content']}], model="gpt-3.5-turbo")

        return True, {"content": response}

max_turn = 1
student_agent = trace(StudentAgent)(seed=13)
user_agent = trace(LLFBenchUserAgent)(env_name="llf-poem-Tanka-v0",
                                      llm_config={"temperature": 0.0, "config_list": config_list})

# ======= Now with the env reward, we can optimize =======

init_obs = user_agent.get_starting_message()
optimizer = LLMOptimizer(student_agent.parameters,
                         config_list=config_list,
                         task_description=dedent("""
                         You are helping a student write a poem that satisfies the following requirements:
                         {}
                         """.format(init_obs)))  # This just concatenates the feedback into the parameter

performances = []
exp_runs = 5

for _ in range(exp_runs):
    optimization_steps = 4
    performance = []

    for _ in range(optimization_steps):
        print("Old prompt:", student_agent.parameters[0].data)

        init_obs = user_agent.get_starting_message()
        user_agent.initiate_chat(student_agent, message=init_obs)
        feedback = user_agent.last_message().data['content']

        performance.append(user_agent.reward_history[-1])

        if user_agent.reward_history[-1] == 1.0:
            print("Reached highest reward.")
            break

        last_message = student_agent.last_message()

        optimizer.zero_feedback()
        last_message.backward(feedback, PropagateStrategy.retain_last_only_propagate, retain_graph=False)
        optimizer.step()

        print("New prompt:", student_agent.parameters[0].data)

    print("Agent reward history:", performance)

    performances.append(performance)

def backfill_lists(parent_list):
    max_length = max(len(child) for child in parent_list)

    for child in parent_list:
        # While the child list is shorter than the longest, append its last element
        while len(child) < max_length:
            child.append(child[-1])

    return parent_list

performances = backfill_lists(performances)

import matplotlib.pyplot as plt
import numpy as np

performances = np.array(performances)

# Calculate mean and standard deviation
means = np.mean(performances, axis=0)
stds = np.std(performances, axis=0)

# Epochs
epochs = np.arange(1, len(means) + 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, means, label='Mean Performance')
plt.fill_between(epochs, means - stds, means + stds, alpha=0.2)

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.title('Performance Across Epochs with Confidence Interval')
plt.legend()
plt.grid(True)

# Show plot
plt.show()