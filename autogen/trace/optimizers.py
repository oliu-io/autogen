from autogen.trace.nodes import ParameterNode
from collections import defaultdict
from autogen import AssistantAgent
from autogen.oai.completion import Completion
from textwrap import dedent, indent
from copy import copy
from autogen.trace.propagators import get_label, get_name

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
            p._feedback = defaultdict(list)


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

class AgentExecutionSummary:
    ANALYSIS_PROMPT = dedent("""
    Considering the following task:

    TASK: {task}
    
    Here are the list of agents involved in completing the task:
    
    AGENT LIST:
    {agent_list}
    
    Here is a brief summary of how these agents attempted to solve this task:
    
    """)


# This updates feedback before the optimizer
class LLMOptimizationPathSummary:
    def __init__(self, parameters, config_list, *args, **kwargs):
        # can add task description in here
        assert type(parameters) is list
        assert all([isinstance(p, ParameterNode) for p in parameters])
        self.parameters = parameters
        sys_msg = dedent("""
                You are analyzing the crash report of an intelligent assistant.
                The crash report prints out the path of the execution of the assistant, but might not be helpful.
                It also contains the feedback the assistant received from the user.
                Give a high-level, concise summary of the crash report.
                """)
        self.llm = AssistantAgent(name="assistant",
                                  system_message=sys_msg,
                                  llm_config={"config_list": config_list})
        self.parameter_copy = {}
        self.save_parameter_copy()

    def save_parameter_copy(self):
        # this is to be able to go back
        for p in self.parameters:
            if p.trainable:
                self.parameter_copy[p] = {"_data": copy(p._data), "_feedback": copy(p._feedback)}

    def update_feedback(self):
        # before update,  we save it once
        self.save_parameter_copy()
        for p in self.parameters:
            if p.trainable:
                p._feedback = self._update_feedback(p._feedback)  # NOTE: This is an in-place update

    def _update_feedback(self, feedback):
        new_feedback_dict = {}
        for parent_node, list_of_feedback in feedback.items():
            all_feedback = "\n\n".join(list_of_feedback)
            messages = [{'content': all_feedback, 'role': 'user'}]
            response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
            # new_feedback = self.llm.client.extract_text_or_function_call(response)[0]
            # new_feedback = Completion.extract_text_or_function_call(response)[0]
            new_feedback = self.llm.client.extract_text_or_completion_object(response)[0]
            print(new_feedback)
            new_feedback_dict[parent_node] = new_feedback
        return new_feedback_dict

class FeedbackEnhance:
    def __init__(self, parameters, config_list, *args, **kwargs):
        # can add task description in here
        assert type(parameters) is list
        assert all([isinstance(p, ParameterNode) for p in parameters])
        self.parameters = parameters
        sys_msg = dedent("""
                ...
        """)
        self.llm = AssistantAgent(name="assistant",
                                  system_message=sys_msg,
                                  llm_config={"config_list": config_list})

    def update_feedback(self):
        for p in self.parameters:
            if p.trainable:
                p._feedback = self._update_feedback(p._feedback)  # NOTE: This is an in-place update

    def _update_feedback(self, feedback):
        new_feedback_dict = {}
        for parent_node, list_of_feedback in feedback.items():
            all_feedback = "\n\n".join(list_of_feedback)
            messages = [{'content': all_feedback, 'role': 'user'}]
            response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
            new_feedback = self.llm.client.extract_text_or_function_call(response)[0]
            new_feedback_dict[parent_node] = new_feedback
        return new_feedback_dict