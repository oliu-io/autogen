from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.trace.trace import trace
import copy


# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
assistant = trace(AssistantAgent)("assistant", llm_config={"config_list": config_list})
user_proxy = trace(UserProxyAgent)("user_proxy", code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")


## A simple demonstration of using backward and optimizer
from autogen.trace.optimizers import DummyOptimizer

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

from graphviz import Digraph


def back_prop_node_visualization(start_node):
    dot = Digraph()
    node = start_node

    visited = set()
    stack = [start_node]

    # we do two loops because I worry Digraph requires "pre-registration" of all nodes
    # add node names
    while stack:
        current_node = stack.pop()
        print(f"Node {node.name}: Node Type {node}, Node: {node._data}")
        if current_node not in visited:
            dot.node(node.name.replace(":", ""), node.name.replace(":", ""))
            visited.add(current_node)
            stack.extend(current_node.parents)

    # add node edges
    visited = set()
    stack = [start_node]

    while stack:
        current_node = stack.pop()
        if current_node not in visited:
            for parent in current_node.parents:
                dot.edge(current_node.name.replace(":", ""), parent.name.replace(":", ""))
            visited.add(current_node)
            stack.extend(current_node.parents)

    return dot


dot = back_prop_node_visualization(last_message)

print(dot.source)
dot.view()
