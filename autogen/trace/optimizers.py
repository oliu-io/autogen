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
        Your goal is to guess what this module is doing and summarize its functionality.
        """)

        super().__init__(config_list, sys_msg, *args, **kwargs)

        execution_prompt = dedent("""            
            <Report>
            {{#each blocks}}
            {{this.text}}

            {{~/each}}
            </Report>
        """)

        summary_prompt = dedent("""
            Here is the task description:
            <Task>
            {{task}}
            </Task>
        
            This is the execution report of this module:
            
            <Report>
            {{#each blocks}}
            {{this.text}}

            {{~/each}}
            </Report>
            
            Please summarize the functionality of this module below:
        """)

        block_prompt = dedent("""
            Inputs: {{input_llms}} -> {{name}}
            {{#each inputs}}
            Input{{this.num}} {{this.llm}}:
                {{this.inpt}}

            {{~/each}}

            Output {{name}}:
            {{output}}
        """)

        self.summary_parser = SimplePromptParser(summary_prompt)
        self.block_parser = SimplePromptParser(block_prompt)
        self.execution_parser = SimplePromptParser(execution_prompt)

    def __call__(self, task_desc, blocks):
        """
        Args:
            blocks: it's a list of blocks because we can run multiple trials of the same module
                    and have all history.
        Returns: str
        """

        # ready to handle multiple blocks, but for on_agent, we might want a different prompt format..
        # or unify names of the blocks
        block_texts = []
        for block in blocks:
            inputs = [{"llm": llm_name, "inpt": inpt, 'num': str(i + 1)} for i, (llm_name, inpt) in
                      enumerate(zip(block.input_llms, block.inputs))]
            block_text = \
                self.block_parser(input_llms=str(block.input_llms), name=block.name, output=block.output,
                                  inputs=inputs)[0]['content']
            block_texts.append(block_text)

        messages = self.summary_parser(blocks=[{'text': b} for b in block_texts], task=task_desc)

        response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
        summary = self.llm.client.extract_text_or_completion_object(response)[0]

        execution_report = self.execution_parser(blocks=[{'text': b} for b in block_texts])[0]['content']

        return execution_report, summary


class LLMModuleVerifier(LLMCallable):
    def __init__(self, config_list, *args, **kwargs):
        sys_msg = dedent("""
        You are given a report of a module's functionality with a list of inputs and the module's output.
        The module is supposed to process or alter the input to fulfill a functionality.
        Your job is to be a harsh reviewer to see if the module is doing what it's supposed to do.
        """)

        super().__init__(config_list, sys_msg, *args, **kwargs)

        review_prompt = dedent("""
        Here is the task description:
        <Task>
        {{task}}
        </Task>
        
        This is the execution report of this module:
        
        <Report>
        {{report}}
        </Report>

        Here is the summary of the functionality of this module:
        <Summary>
        {{summary}}
        </Summary>
        
        Please judge if this module successfully fulfilled its designed intentions.
        One important question is -- did it change the input? How much did it change the input?
        
        Write your judgment below -- be succinct:
        """)
        self.review_parser = SimplePromptParser(review_prompt)

    def __call__(self, task_desc, execution_report, summary):
        """
        Args:
            execution_report: should come from LLMModuleSummarizer
            summary: should come from LLMModuleSummarizer
        """
        messages = self.review_parser(task=task_desc, report=execution_report, summary=summary)
        response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
        return self.llm.client.extract_text_or_completion_object(response)[0]

# causal attribution
# blame assignment

# in order to do system updates, we need:
# delta x, delta y
# delta y is from feedback, not much we can do or need to do (besides synthesis)
# delta x is from input, system analysis
class LLMAgentGraphDesigner(LLMCallable):
    pass


if __name__ == '__main__':
    # add a few unit tests here
    pass
