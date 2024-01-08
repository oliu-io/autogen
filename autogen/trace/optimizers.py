from autogen.trace.nodes import ParameterNode
from collections import defaultdict
from autogen import AssistantAgent
from textwrap import dedent, indent


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
        new_instruct = self.llm.client.extract_text_or_function_call(response)[0]
        new_node = {'content': new_instruct, 'role': 'system'}
        return new_node


class PropagateStrategy:
    @staticmethod
    def retain_last_only_propagate(child):
        summary = ''.join(
            [v[0] for k, v in child.feedback.items()])
        return {parent: summary for parent in child.parents}

    @staticmethod
    def retain_full_history_propagate(child):
        # this retains the full history
        summary = ''.join([f'{str(k)}:{v[0]}' for k, v in
                           child.feedback.items()])
        return {parent: summary for parent in child.parents}


# This updates feedback before the optimizer
class FeedbackEnhancer:
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
            print(new_feedback)
            new_feedback_dict[parent_node] = new_feedback
        return new_feedback_dict
