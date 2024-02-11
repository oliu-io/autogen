"""
In this file, we should have:
2. Add FeedbackEnhance
3. An optimizer that optimizes...with an agent call (20 min)
"""

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, Agent
from autogen.trace.trace import trace, compatability, node, trace_class
from autogen.trace.optimizers import PropagateStrategy, LLMOptimizer
from textwrap import dedent, indent
from env_wrapper import LLFBenchUserAgent

from autogen.trace.optimizers import DummyOptimizer

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict={
             "model": ["gpt-3.5-turbo-0613", "gpt-3.5-turbo"],
         })
assert len(config_list) > 0

termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

sys_msg1 = dedent("You are a student and your teacher gives you an assignment to write a poem.")

class PoemStudentAgent(AssistantAgent):

    def __init__(self):
        super().__init__(
            name="PoemStudentAgent",
            system_message=sys_msg1,
            llm_config={"temperature": 0.0, "config_list": config_list},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
        )

sys_msg2 = dedent("You are extracting a poem from the student's message. " +
                 "Do not extract anything except the poem itself."
                 "If the student did not write a poem, return an empty string.")
class PoemExtractor(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="PoemExtractor",
            system_message=sys_msg2,
            llm_config={"temperature": 0.0, "config_list": config_list},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
        )

# We inherit from the traced version of AssistantAgent and register new reply_funcs that based on nodes.
# class PoemAgent(trace(AssistantAgent, wrap_all_replies=False)):
@trace_class
class PoemAgent(AssistantAgent):
    def __init__(self, seed=1234):
        super().__init__(
            name="PoemAgent",
            system_message="",
            llm_config={"temperature": 0.0, "config_list": config_list, 'cache_seed': seed},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
            human_input_mode= "NEVER"
        )
        self.student_agent = trace(PoemStudentAgent)()
        self.extractor_agent = trace(PoemExtractor)()

        self.poem = None

        self.register_reply(UserProxyAgent, PoemAgent._generate_poem_reply, position=1)
        self.register_reply([PoemStudentAgent], PoemAgent._reply_to_terminate_agent)
        self.register_reply([PoemExtractor], PoemAgent._reply_to_terminate_extractor)

        # self.stop_reply_at_receive(self.student_agent)
        # self.stop_reply_at_receive(self.extractor_agent)

    def get_last_user_message(self, agent):
        for m in reversed(self.chat_message_nodes[agent]):
            if m['role'] == 'user':
                return m

    def _generate_poem_reply(self,
        messages = None, sender=None, config=None):
        # message = messages[-1]['content']
        message = messages[-1]

        if self.poem is None:
            self.initiate_chat(self.student_agent, message=message, clear_history=True)
            self.poem = self.get_last_user_message(self.student_agent)#["content"]

        # this just means we haven't called extractor agent before
        if len(self._oai_messages[self.extractor_agent]) == 0:
            self.initiate_chat(self.extractor_agent, message=self.poem, clear_history=True)

        # extracted_poem = self.get_last_user_message(self.extractor_agent)["content"]
        extracted_poem = self.get_last_user_message(self.extractor_agent)#["content"]

        return True, extracted_poem# {"content": extracted_poem}

    def _reply_to_terminate_agent(self, messages=None, sender=None, config=None):
        return True, node({"content": "TERMINATE"})

    def _reply_to_terminate_extractor(self, messages=None, sender=None, config=None):
        return True, node({"content": "TERMINATE"})

max_turn = 1
poem_agent = PoemAgent(seed=13)

user_agent = trace(LLFBenchUserAgent)(env_name="llf-poem-SyllableConstrainedPoem-v0",
                                      llm_config={"temperature": 0.0, "config_list": config_list})

# ======= Now with the env reward, we can optimize =======

init_obs = user_agent.get_starting_message()
optimizer = LLMOptimizer(poem_agent.parameters,
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
        print("Old prompt:", poem_agent.parameters[0].data)

        init_obs = user_agent.get_starting_message()
        user_agent.initiate_chat(poem_agent, message=init_obs, clear_history=True)
        feedback = user_agent.last_message_node().data['content']

        performance.append(user_agent.reward_history[-1])

        if user_agent.reward_history[-1] == 1.0:
            print("Reached highest reward.")
            continue

        # last_message = poem_agent.last_message_node()
        last_message = poem_agent.last_message_node(user_agent, role='assistant')

        optimizer.zero_feedback()
        last_message.backward(feedback, PropagateStrategy.retain_last_only_propagate, retain_graph=False)
        optimizer.step()

        print("New prompt:", poem_agent.parameters[0].data)

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