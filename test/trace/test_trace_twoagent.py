from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.trace.trace import trace
import copy
from autogen.trace.optimizers.optimizers import DummyOptimizer


# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
assistant = trace(AssistantAgent)("assistant", llm_config={"config_list": config_list})
user_proxy = trace(UserProxyAgent)("user_proxy", code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")


## A simple demonstration of using backward and optimizer

optimizer = DummyOptimizer(assistant.parameters)  # This just concatenates the feedback into the parameter


def propagate(child):
    # a dummy function for testing
    summary = "".join(
        [f"{str(k)}:{v[0]}" for k, v in child.feedback.items()]
    )  # we only take the first feedback for testing purposes
    return {parent: summary for parent in child.parents}


feedback = "Great job."
last_message = assistant.last_message_node()
optimizer.zero_feedback()
last_message.backward(feedback, propagate, retain_graph=True)  # Set retain_graph for testing
optimizer.step()

# Test check a path from output to input
assert feedback in optimizer.parameters[0]
assert all(
    [feedback in v[0] for v in optimizer.parameters[0].feedback.values()]
)  # make sure feedback is propagated to the parameter
node = last_message
while True:
    assert all([feedback in v[0] for v in node.feedback.values()])
    # print(f'Node {node.name} at level {node.level}: value {node.data} Feedback {node.feedback}')
    print(f"Node {node.name} at level {node.level}: Feedback {node.feedback}")

    if len(node.parents) > 0:
        node = node.parents[0]
    else:
        break
