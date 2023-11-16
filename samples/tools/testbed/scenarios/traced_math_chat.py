from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.trace.trace import trace, compatability
import copy
import random
#from ...include import testbed_utils

#testbed_utils.init()
global_seed = 42
random.seed(global_seed)

config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={
        "model": {
            "gpt-4",
            "gpt4",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-4-32k-v0314",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "chatgpt-35-turbo-0301",
            "gpt-35-turbo-v0301",
            "gpt",
        }
    }
)

assistant = trace(AssistantAgent)(
    "assistant",
    system_message="You are a helpful assistant.",
    is_termination_msg=lambda x: x.get("content", "").rstrip().find("TERMINATE") >= 0,
    llm_config={
        "timeout": 600,
        "seed": global_seed,
        "config_list": config_list,
    },
)

# If you do not want to trace the mathproxyagent, you can use the following line instead.
# Beware that the backward and optimizer methods further below may not work properly if you do not trace the mathproxyagent.
#mathproxyagent = compatability(MathUserProxyAgent)(
mathproxyagent = trace(MathUserProxyAgent)(
    "mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False,
    },
    #max_consecutive_auto_reply=10,
    #default_auto_reply="TERMINATE",
)

# given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial message.
# the assistant receives the message and generates a response. The response will be sent back to the mathproxyagent for processing.
# The conversation continues until the termination condition is met, in MathChat, the termination condition is the detect of "\boxed{}" in the response.
math_problem = "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation."

import datasets
data = datasets.load_dataset("competition_math")
train_data = data["train"].shuffle(seed=global_seed)
test_data = data["test"].shuffle(seed=global_seed)
train_data = [
    {
        "problem": train_data[x]["problem"],
        "solution": train_data[x]["solution"],
    }
    for x in range(len(train_data)) if train_data[x]["level"] == "Level 5" and train_data[x]["type"] != "Geometry"
]
test_data = [
    {
        "problem": test_data[x]["problem"],
        "solution": test_data[x]["solution"],
    }
    for x in range(len(test_data)) if test_data[x]["level"] == "Level 5" and test_data[x]["type"] != "Geometry"
]
print(len(train_data), len(test_data))

math_problem = train_data[random.randint(0, len(train_data))]["problem"]
mathproxyagent.initiate_chat(assistant, problem=math_problem)


#testbed_utils.finalize(agents=[assistant, mathproxyagent])

## Using backward and optimizer from the trace_twoagent sample code
from autogen.trace.optimizers import DummyOptimizer
optimizer = DummyOptimizer(assistant.parameters)  # This just concatenates the feedback into the parameter
def propagate(child, feedback):
    # a dummy function for testing
    return {parent: copy.copy(feedback) for parent in child.parents}
feedback = 'Great job.'
last_message = assistant.last_message()
optimizer.zero_feedback()
last_message.backward(feedback, propagate, retain_graph=True)  # Set retain_graph for testing
optimizer.step()

# Test check a path from output to input
print()
assert feedback in optimizer.parameters[0]
assert all([v == feedback for v in optimizer.parameters[0]._feedback.values()])  # make sure feedback is propagated to the parameter
node = last_message
while True:
    assert all([v == feedback for v in node._feedback.values()])
    # print(f'Node {node.name} at level {node.level}: value {node.data} Feedback {node._feedback}')
    print(f'Node {node.name} at level {node.level}: Feedback {node._feedback}')

    if len(node.parents)>0:
        node = node.parents[0]
    else:
        break