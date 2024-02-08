import autogen
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, Agent
from autogen.trace.trace import trace, compatability
from textwrap import dedent, indent
from autogen.trace.utils import back_prop_node_visualization

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict={
             "model": ["gpt-3.5-turbo-0613", "gpt-3.5-turbo"],
         })
assert len(config_list) > 0

# 1. create an AssistantAgent instance named "assistant"
assistant = trace(autogen.AssistantAgent)(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

# 2. create the MathUserProxyAgent instance named "mathproxyagent"
# By default, the human_input_mode is "NEVER", which means the agent will not ask for human input.
mathproxyagent = trace(MathUserProxyAgent)(
    name="mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

math_problem = (
    "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation."
)
mathproxyagent.initiate_chat(assistant, problem=math_problem, clear_history=True)

def propagate(child):
    # a dummy function for testing
    summary =''.join([ f'{str(k)}:{v[0]}' for k,v in child.feedback.items()])  # we only take the first feedback for testing purposes
    return {parent: summary for parent in child.parents}

# last_message = assistant._oai_messages[-1]
# print(last_message.data)

last_message = mathproxyagent.last_message_node()
feedback = "The solution is correct."  # imagine we have access to groundtruth answers here

dot = last_message.backward(feedback, propagate, retain_graph=False, visualize=True)

# dot = back_prop_node_visualization(last_message)

# print(dot.source)
dot.view()