"""
In this file, we should have:
1. ~~An Agent that solves Poem (a base agent) (you have this already)~~
  - add trace agent to it
2. A propagate that only gives feedback to prompt node (manually)
   - where would FeedbackEnhance go?
3. An optimizer that optimizes...with an agent call
"""

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, Agent
from autogen.trace.trace import trace
from textwrap import dedent, indent
# from env_wrapper import LLFBenchUserAgent
import llfbench

from autogen.trace.optimizers import DummyOptimizer

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict={
    "model": ["gpt-3.5-turbo-0613", "gpt-3.5-turbo"],
})
assert len(config_list) > 0

termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

sys_msg = dedent("You are a student and your teacher gives you an assignment to write a poem" +
                 "Reply \"TERMINATE\" in the end when everything is done.")

class PoemStudentAgent(AssistantAgent):

    def __init__(self):
        super().__init__(
            name="PoemStudentAgent",
            system_message=sys_msg,
            llm_config={"temperature": 0.0, "config_list": config_list},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
        )


sys_msg = dedent("You are extracting a poem from the student's message. " +
                 "Do not extract anything except the poem itself." +
                 "If the student did not write a poem, return an empty string." +
                 "Reply \"TERMINATE\" in the end when everything is done.")


class PoemExtractor(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="PoemExtractor",
            system_message=sys_msg,
            llm_config={"temperature": 0.0, "config_list": config_list},
            max_consecutive_auto_reply=1,
        )

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

max_turn = 1
poem_agent = PoemStudentAgent()
extractor_agent = PoemExtractor()

env = llfbench.make("llf-poem-Haiku-v0", instruction_type='b', feedback_type='a')
obs, info = env.reset()

# current code won't run because autogen we have does not support this function
chat_results = user.initiate_chats(
    [
        {
            "recipient": poem_agent,
            "message": obs['instruction'],
            "clear_history": True,
            "silent": False,
            "summary_method": "last_msg",
        },
        {
            "recipient": extractor_agent,
            "message": "Extracting the poem.",
            "summary_method": "last_msg",
        }
    ]
)

import pdb; pdb.set_trace()