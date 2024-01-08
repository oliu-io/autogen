"""
In this file, we should have:
2. Add FeedbackEnhance
3. An optimizer that optimizes...with an agent call (20 min)
"""

import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.trace.trace import trace
from textwrap import dedent, indent
from env_wrapper import LLFBenchUserAgent
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

from autogen.trace.optimizers import DummyOptimizer, LLMOptimizer


# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
config_list = [config_list[1]]
assert config_list[0]["model"] == "gpt-3.5-turbo-0613"

termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

sys_msg = dedent("""
You are a helpful assistant.
You extract only lines of poems in the message frmo the student, ignore any part of the message that is not related to the poem.
You should only reply with the poem string extracted from the user's input.
""")
class PoemExtractAgent(AssistantAgent):

    def __init__(self):
        # Modified from BoardAgent in ChessBaord example
        super().__init__(
            name="PoemExtractAgent",
            system_message=sys_msg,
            llm_config={"temperature": 0.0, "config_list": config_list},
            max_consecutive_auto_reply=5
        )
        self.register_reply(autogen.ConversableAgent, PoemExtractAgent._generate_extractor_reply)
        self.correct_messages = defaultdict(list)

    def _generate_extractor_reply(self,
                                  messages: Optional[List[Dict]] = None,
                                  sender: Optional[autogen.Agent] = None,
                                  config: Optional[Any] = None,
                                  ):
        message = messages[-1]
        # extract the poem
        reply = self.generate_reply(self.correct_messages[sender] + [message], sender,
                                    exclude=[PoemExtractAgent._generate_board_reply])
        poem = reply if isinstance(reply, str) else str(reply["content"])
        self.correct_messages[sender].extend([message, self._message_to_dict(poem)])
        self.correct_messages[sender][-1]["role"] = "assistant"
        return True, poem


sys_msg = dedent("You are a student and your teacher gives you an assignment to write a poem.")

class StudentAgent(AssistantAgent):
    def __init__(self, seed=1234, extractor_agent=None):
        super().__init__(
            name="StudentAgent",
            system_message=sys_msg,
            llm_config={"temperature": 0.0, "config_list": config_list, 'cache_seed': seed},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
        )
        self.extractor_agent = extractor_agent
        self.register_reply(autogen.ConversableAgent, StudentAgent._generate_reply_for_user, config=self.extractor_agent)

    def _generate_reply_for_user(self,
                                 messages: Optional[List[Dict]] = None,
                                 sender: Optional[autogen.Agent] = None,
                                 config: Optional[PoemExtractAgent] = None,
                                 ):
        poem_extractor_agent = config
        message = self.generate_reply(messages, sender, exclude=[self._generate_reply_for_user])
        if message is None:
            return True, None
        # if there is a response, we try to extract it
        self.initiate_chat(poem_extractor_agent, clear_history=True, message=message, silent=self.human_input_mode == "NEVER")
        # last message sent by the board agent
        last_message = self._oai_messages[poem_extractor_agent][-1]
        if last_message["role"] == "assistant":
            # I don't know when this will be triggered?
            return True, None

        # I don't know if this is correct either
        return True, self._oai_messages[poem_extractor_agent][-2]


sys_msg = dedent("You are a student and your teacher gives you an assignment to write a poem.")


max_turn = 1
extractor_agent = PoemExtractAgent()
student_agent = trace(StudentAgent)(seed=13, extractor_agent=extractor_agent)
user_agent = trace(LLFBenchUserAgent)(env_name="llf-poem-Haiku-v0",
                                      llm_config={"temperature": 0.0, "config_list": config_list})

init_obs = user_agent.get_starting_message()

# ======= Now with the env reward, we can optimize =======

optimizer = LLMOptimizer(student_agent.parameters,
                         config_list=config_list,
                         task_description=dedent("""
                         You are helping a student write a poem that satisfies the following requirements:
                         {}
                         """.format(init_obs)))  # This just concatenates the feedback into the parameter


# use closure to safely add an agent...
def info_propagate(info_merge_agent=None):
    def propagate(child):
        # we only take the actual feedback, no concat
        summary = ''.join(
            [v[0] for k, v in child.feedback.items()])  # we only take the first feedback for testing purposes
        return {parent: summary for parent in child.parents}

    return propagate


propagate = info_propagate(None)

print("Old prompt:", student_agent.parameters[0].data)

user_agent.initiate_chat(student_agent, message=init_obs)
feedback = user_agent.last_message().data['content']
last_message = student_agent.last_message()
optimizer.zero_feedback()
last_message.backward(feedback, propagate, retain_graph=False)  # Set retain_graph for testing
optimizer.step()

print("New prompt:", student_agent.parameters[0].data)

user_agent.initiate_chat(student_agent, message=init_obs)
feedback = user_agent.last_message().data['content']
last_message = student_agent.last_message()
optimizer.zero_feedback()
last_message.backward(feedback, propagate, retain_graph=False)  # Set retain_graph for testing
optimizer.step()

print("New prompt:", student_agent.parameters[0].data)
