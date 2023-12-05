"""
In this file, we should have:
2. Add FeedbackEnhance
3. An optimizer that optimizes...with an agent call (20 min)
"""

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.trace.trace import trace
from textwrap import dedent, indent
from env_wrapper import VerbalGymUserAgent

from autogen.trace.optimizers import DummyOptimizer, LLMOptimizer

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
config_list = [config_list[1]]
assert config_list[0]["model"] == "gpt-3.5-turbo-0613"

termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


# TODO: like the chessboard, PoemAgent needs to call this before
class PoemExtractAgent(AssistantAgent):

    def __init__(self):
        super().__init__(
            name="PoemExtractAgent",
            system_message=sys_msg,
            llm_config={"temperature": 0.0, "config_list": config_list},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
        )


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


max_turn = 1
student_agent = trace(StudentAgent)(seed=13)
user_agent = trace(VerbalGymUserAgent)(env_name="verbal-poem-Haiku-v0",
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
    # def propagate(child):
    #     # a dummy function for testing
    #     summary = ''.join([f'{str(k)}:{v[0]}' for k, v in
    #                        child.feedback.items()])  # we only take the first feedback for testing purposes
    #     return {parent: summary for parent in child.parents}
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
last_message.backward(feedback, propagate, retain_graph=True)  # Set retain_graph for testing
optimizer.step()

print("New prompt:", student_agent.parameters[0].data)

user_agent.initiate_chat(student_agent, message=init_obs)
feedback = user_agent.last_message().data['content']
last_message = student_agent.last_message()
optimizer.zero_feedback()
last_message.backward(feedback, propagate, retain_graph=True)  # Set retain_graph for testing
optimizer.step()

print("New prompt:", student_agent.parameters[0].data)