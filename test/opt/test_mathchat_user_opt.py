"""
This file demonstrates how to do a fixed dataset based optimization
(Other files focus on interactive environments)

Example notebook: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_MathChat.ipynb

Example notebook: https://github.com/microsoft/autogen/blob/main/notebook/oai_chatgpt_gpt4.ipynb
"""

import autogen
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, Agent
from autogen.trace.trace import trace, compatibility
from autogen.trace.optimizers import TeacherLLMOptimizer
from autogen.trace.propagators import retain_last_only_propagate
from textwrap import dedent, indent
from autogen.trace.optimizer_autogen import train_with_datasets, DatasetProcessor

from autogen.trace.utils import backfill_lists, plot_agent_performance, verbalize

from autogen.code_utils import UNKNOWN, extract_code, execute_code, infer_lang

# pip install datasets
from datasets import load_dataset

from autogen.math_utils import is_equiv

dataset = load_dataset("hendrycks/competition_math")

config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-3.5-turbo-0613", "gpt-3.5-turbo"],
    },
)
assert len(config_list) > 0

# ====== Helper Functions ======


def remove_boxed(string: str):
    """Source: https://github.com/hendrycks/math
    Extract the text within a \\boxed{...} environment.
    Example:

    > remove_boxed("\\boxed{\\frac{2}{3}}")

    \\frac{2}{3}
    """
    left = "\\boxed{"
    if left not in string:
        left = "\boxed{"
    try:
        if not all((string[: len(left)] == left, string[-1] == "}")):
            raise AssertionError

        return string[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string: str):
    """Source: https://github.com/hendrycks/math
    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            idx = string.rfind("\boxed")
            if idx < 0:
                return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def get_answer(solution):
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    if answer is None:
        return None
    return answer


def _is_termination_msg_mathchat(message):
    """Check if a message is a termination message."""
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    for c in cb:
        if c[0] == "python" or c[0] == "wolfram":
            break

    terminate = get_answer(message) is not None and get_answer(message) != ""

    return terminate


# 1. create an AssistantAgent instance named "assistant"
assistant = trace(autogen.AssistantAgent)(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
    max_consecutive_auto_reply=2,
)

# 2. create the MathUserProxyAgent instance named "mathproxyagent"
# By default, the human_input_mode is "NEVER", which means the agent will not ask for human input.
mathproxyagent = trace(MathUserProxyAgent)(
    name="mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    is_termination_msg=_is_termination_msg_mathchat,
)


def generate_answer(math_problem):
    mathproxyagent.initiate_chat(assistant, problem=math_problem, clear_history=True)
    last_message = mathproxyagent.last_message_node(role="user")
    # "assistant" means self, "user" means the other agent
    return last_message


def check_equiv(sol1, sol2):
    return is_equiv(get_answer(sol1), get_answer(sol2))


dp = DatasetProcessor("problem", "solution", reward_fn=check_equiv)

optimizer = TeacherLLMOptimizer(
    assistant.parameters,
    config_list=config_list,
    task_description=dedent(
        """
         You are helping a student solve math problems.
         Give them some instructions on how to avoid errors.
         """
    ),
)

performances = []
exp_runs = 1
optimization_steps = 1

for _ in range(exp_runs):
    info = train_with_datasets(
        dataset["train"],
        dp,
        generate_answer,
        optimizer,
        steps=optimization_steps,
        propagate_fn=retain_last_only_propagate(),
        verbose=False,
        seed=123,
    )
    print("Agent reward history:", info["rewards"])
    performances.append(info["rewards"])

performances = backfill_lists(performances)
plot_agent_performance(performances, backfilled=True)
