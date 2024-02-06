from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.trace.trace import trace
import copy

import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.trace.trace import trace, node, Node
from autogen.trace.utils import back_prop_node_visualization
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
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
config_list = [config_list[1]]
assert config_list[0]["model"] == "gpt-3.5-turbo-0613"

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

init_obs = user_agent.get_starting_message()
user_agent.initiate_chat(student_agent, message=init_obs)

def propagate(child):
    # a dummy function for testing
    print("called")
    summary =''.join([ f'{str(k)}:{v[0]}' for k,v in child.feedback.items()])  # we only take the first feedback for testing purposes
    return {parent: summary for parent in child.parents}

last_message = student_agent.last_message()
feedback = user_agent.last_message().data['content']

last_message.backward(feedback, propagate, retain_graph=False)

dot = back_prop_node_visualization(last_message)

print(dot.source)
dot.view()