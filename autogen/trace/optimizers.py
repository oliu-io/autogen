from autogen.trace.nodes import ParameterNode
from collections import defaultdict
from autogen import AssistantAgent
from autogen.oai.completion import Completion
from textwrap import dedent, indent
from copy import copy
from autogen.trace.propagators import get_label, get_name
from dataclasses import dataclass
from autogen.trace.utils import SimplePromptParser

"""
We follow the same design principle as trace
This file is not dependent of AutoGen library and can be used independently with trace
"""


class Optimizer:

    def __init__(self, parameters, *args, **kwargs):
        assert type(parameters) is list
        assert all([isinstance(p, ParameterNode) for p in parameters])
        self.parameters = parameters

    def step(self):
        for p in self.parameters:
            if p.trainable:
                p._data = self._step(p._data, p._feedback)  # NOTE: This is an in-place update

    def _step(self, value, feedback):
        raise NotImplementedError

    def zero_feedback(self):
        for p in self.parameters:
            # trace the children of parameters and clean their feedback
            p.zero_feedback()


class DummyOptimizer(Optimizer):
    # FOR TESTING PURPOSES ONLY

    def __init__(self, parameters, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)

    def _step(self, value, feedback):

        if isinstance(value, dict):
            base = value['content']
        elif isinstance(value, str):
            base = value
        else:
            raise NotImplementedError
        new = base + ' '.join([' '.join(l) for l in feedback.values()])
        return new


class LLMOptimizer(Optimizer):
    def __init__(self, parameters, config_list, task_description, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        sys_msg = dedent("""
        You are giving instructions to a student on how to accomplish a task.
        The student aims to get a high score.
        Given the feedback the student has received and the instruction you have given,
        You want to come up with a new instruction that will help the student get a higher score.
        """)
        self.llm = AssistantAgent(name="assistant",
                                  system_message=sys_msg,
                                  llm_config={"config_list": config_list})
        self.task_description = task_description

        self.parameter_copy = {}
        self.save_parameter_copy()

    def _step(self, value, feedback):
        # `prompt="Complete the following sentence: {prefix}, context={"prefix": "Today I feel"}`
        # context = {"task_description": self.task_description,
        #            "instruction": value,
        #            "feedback": feedback}
        context = [self.task_description, value['content'], feedback]
        prompt_space = dedent("""
        {}

        This is your instruction to the student from the previous round:
        {}

        Using this instruction, this is the feedback the student received:
        {}

        Please write down a new instruction to help the student achieve a higher score.
        Be concise and to the point.
        Remember the student is starting from scratch, not revising their old work:
        """.format(*context))

        messages = [{'content': prompt_space, 'role': 'user'}]
        response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
        # response = Completion.create(messages=self.llm._oai_system_message + messages)
        # new_instruct = self.llm.client.extract_text_or_function_call(response)[0]
        new_instruct = self.llm.client.extract_text_or_completion_object(response)[0]
        new_node = {'content': new_instruct, 'role': 'system'}
        return new_node

    def save_parameter_copy(self):
        # this is to be able to go back
        for p in self.parameters:
            if p.trainable:
                self.parameter_copy[p] = {"_data": copy(p._data), "_feedback": copy(p._feedback)}

    def zero_feedback(self):
        # if we are ready to perform another round of update
        # we save the current parameters
        self.save_parameter_copy()
        super().zero_feedback()

    def restore_parameters(self, parameters):
        # revert back to a saved copy
        for p in self.parameters:
            if p.trainable:
                assert p in self.parameter_copy
                p._data = self.parameter_copy[p]["_data"]
                p._feedback = self.parameter_copy[p]["_feedback"]


@dataclass
class Block:
    name: str
    output: str
    input_llms: list
    output_llms: list
    inputs: list

    def __str__(self):
        return ""

    def __eq__(self, other):
        # just check the LLM name
        # and content
        return self.name == other.name and self.output == other.output


def construct_block(block_str: str, preserve_content_format=True):
    input_llm, output_llm = "", ""
    llm_name_to_cnt = {}

    sep = '\n' if preserve_content_format else ''

    llm_name_continued = None
    for line in block_str.split("\n"):
        if "->" in line:
            input_llm, output_llm = line.split(" -> ")
            input_llm, output_llm = input_llm.strip("\""), output_llm.strip("\"")
            continue
        if "label=" in line:
            name = line.split(" [label=")[0].strip().strip("\"")
            llm_name_to_cnt[name] = []
            llm_name_continued = name
            continue

        llm_name_to_cnt[llm_name_continued].append(line.rstrip("\"]"))

    return Block(name=output_llm, output=sep.join(llm_name_to_cnt[output_llm]), input_llms=[input_llm], output_llms=[],
                 inputs=[sep.join(llm_name_to_cnt[input_llm])])

def parse_blocks(dot_str, preserve_content_format=True):
    # dot_str = digraph.source

    blocks = []
    block_continued = []
    for i, line in enumerate(dot_str.split('\n')):
        if "->" in line and len(block_continued) != 0:
            blocks.append("\n".join(block_continued))
            block_continued = [line.strip()]
        elif "}" not in line and line.strip() != "" and "{" not in line:
            block_continued.append(line.strip())  # give back the line break

    blocks.append("\n".join(block_continued))

    constructed_blocks = []

    for block in blocks:
        # block = [input -> output, input_llm_str, output_llm_str]
        block_rep = construct_block(block, preserve_content_format)
        constructed_blocks.append(block_rep)

    # re-assemble:
    # rule: if input -> output, if ouput is the same, add the input to the input list (merge)
    # the current merge is on LLM name + function name
    # but we might want to perform this on the LLM agent level

    # input -> output
    # if the node name is in another llm's input list, then, we add that node to the output llm list
    finished_blocks = []
    processed_blocks = copy(constructed_blocks)
    while len(processed_blocks) != 0:
        block = processed_blocks.pop()

        if block in finished_blocks:
            continue

        # print(f"processing {block.name}")
        # add input llms
        same_blocks = filter(lambda x: x == block, processed_blocks)
        for b in same_blocks:
            # print(b.name)
            # print(b.input_llms)
            block.input_llms.extend(b.input_llms)
            block.inputs.extend(b.inputs)

        # add output llms
        outputs = filter(lambda x: block.name in x.input_llms, processed_blocks)
        out_llm_names = [o.name for o in outputs]
        block.output_llms.extend(out_llm_names)

        finished_blocks.append(block)

    return finished_blocks


def display_full_graph(blocks):
    assert type(blocks[0]) == Block
    content = ""

    # we traverse each block
    # we start by nodes that have the most input llms that have no parents
    sorting_list = []
    block_names = [b.name for b in blocks]
    for b in blocks:
        cnt = 0
        for i in b.input_llms:
            if i in block_names:
                cnt += 1  # it has an input LLM that's also dependent on other LLMs
        sorting_list.append((b, cnt))
    sorting_list = sorted(sorting_list, key=lambda x: x[1])
    blocks = [b[0] for b in sorting_list]

    # print input one by one, and then output
    for b in blocks:
        # input llms -> output llm
        content += f"Inputs: {b.input_llms} -> {b.name}\n"
        for i, iput in enumerate(b.inputs):
            content += f"  Input {b.input_llms[i]}:\n"
            content += f"    {iput}\n"

        content += f"Output: {b.name}\n"
        content += f"    {b.output}\n"

        content += "\n"

    return content


class LLMCallable(object):
    def __init__(self, config_list, system_message,
                 prompt_template,
                 name="helper"):
        self.config_list = config_list
        self.llm = AssistantAgent(name=name,
                                  system_message=system_message,
                                  llm_config={"config_list": config_list})
        self.parser = SimplePromptParser(prompt_template, verbose=False)

    def create_simple_response(self, message):
        messages = [{'content': message, 'role': 'user'}]
        response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
        return self.llm.client.extract_text_or_completion_object(response)[0]

    def create_response(self, **kwargs):
        results = self.parser(**kwargs)
        messages = [{'content': results, 'role': 'user'}]
        response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
        return self.llm.client.extract_text_or_completion_object(response)[0]


class LLMModuleSummarizer(LLMCallable):
    def __init__(self, config_list, *args, **kwargs):
        sys_msg = dedent("""
        You are given an analysis report of a module with a list of inputs and the module's output.
        The module can take many inputs and respond to inputs with a single output.
        Your goal is to guess what this module is doing and summarize it's functionality.
        """)

        super().__init__(config_list, *args, **kwargs)

    def __call__(self, blocks):
        """
        Args:
            blocks: it's a list of blocks because we might want to summarize the functionality of an entire agent
                    not just for a sub-function. The list of blocks should be all the functions this agent is asked to provide.

        Returns: str
        """
        # TODO: 1. if it's on agent, we reduce the name/change rep
        on_agent = False
        if len(blocks) > 1:
            on_agent = True
        assert not on_agent, "Not implemented yet"

        prompt_space = dedent("""
        This is the execution report of this module:
        """)

        messages = [{'content': prompt_space, 'role': 'user'}]
        response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
        new_instruct = self.llm.client.extract_text_or_completion_object(response)[0]
        new_node = {'content': new_instruct, 'role': 'system'}
        return new_node


class AgentExecutionSummary:
    ANALYSIS_PROMPT = dedent("""
    Considering the following task:

    TASK: {task}

    Here are the list of agents involved in completing the task:

    AGENT LIST:
    {agent_list}

    Here is a brief summary of how these agents attempted to solve this task:

    """)


def simple_shrink(dot_str, shrink=True):
    """
    This provides a heuristic shrink to reduce the lines of docstring that describes the graph.

    Args:s
        dot_str: the returned object from calling backward on a node with (visualize=True, reverse_plot=False)
        shrink: if set True, the dot_str will not be a valid GraphViz dot str; otherwise it will still be valid

    Returns:
        A string representation of the graph

    """

    begin_str = """digraph {""" + '\n'
    end_str = """}"""

    # step 1: break into miniblocks
    blocks = []
    block_continued = []
    for i, line in enumerate(dot_str.split('\n')):
        if "->" in line and len(block_continued) != 0:
            blocks.append("\n".join(block_continued))
            block_continued = [line]
        elif "}" not in line and line.strip() != "" and "{" not in line:
            block_continued.append(line)  # give back the line break

    blocks.append('\n'.join(block_continued))

    # step 2: re-order blocks based on "->" directions
    sorted_blocks = []

    for block in blocks:
        sub_blocks = []  # should have 3 elements
        continued_sub = []
        for b in block.split('\n'):
            if '->' in b:
                sub_blocks.append(b)
            elif b.strip()[-1] == ']' and len(continued_sub) != 0:
                continued_sub.append(b)
                sub_blocks.append("\n".join(continued_sub))
                continued_sub = []
            else:
                continued_sub.append(b)

        # check order now
        ordered_sub_blocks = [sub_blocks[0]]
        first, second = sub_blocks[0].strip().split(" -> ")

        if first in sub_blocks[1] and second in sub_blocks[2]:
            ordered_sub_blocks.extend([sub_blocks[1], sub_blocks[2]])
        else:
            ordered_sub_blocks.extend([sub_blocks[2], sub_blocks[1]])

        sorted_blocks.append(ordered_sub_blocks)

    blocks = sorted_blocks
    # reverse the block to reveal computation structure from top to bottom
    blocks.reverse()

    # step 3: shrink the str representation by the following ops:
    # (all of these are inspired by Graphviz's actual display)
    # By default, we only want to display the message sender (parent's node message)
    # We only display child when remote edges occur

    # 3.1 - if a node has multiple parents, we don't display the child node's content until after displaying all the parents
    # 3.2 - if a child node is immeidately the parent of another node

    shrunk_blocks = []
    for i in range(len(sorted_blocks)):
        # forward search to find common child
        child = sorted_blocks[i][0].strip().split(" -> ")[1]
        found = False
        # condition 1: look-ahead (if the child has multiple parents, we delay till the last parent)
        for block in sorted_blocks[i + 1:]:
            if child in block[0]:
                # see if it's "-> child" or "child ->"
                left, right = block[0].strip().split(" -> ")
                if right == child:
                    found = True
        # condition 2: if the next immediate step, the child is a message sender, then it will be displayed anyway, we can skip here
        if i + 1 < len(sorted_blocks):
            left, right = sorted_blocks[i + 1][0].strip().split(" -> ")
            if left == child:
                found = True

        if found:
            shrunk_blocks.append(sorted_blocks[i][:2])
        else:
            shrunk_blocks.append(sorted_blocks[i])

    blocks = shrunk_blocks
    blocks = ["\n".join(b) for b in blocks]

    return begin_str + "\n".join(blocks) + '\n' + end_str
