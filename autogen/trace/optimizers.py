from typing import Any, List, Dict
from autogen.trace.nodes import ParameterNode, Node
from collections import defaultdict
from autogen import AssistantAgent
from autogen.oai.completion import Completion
from textwrap import dedent, indent
from copy import copy
from autogen.trace.propagators import Propagator, FunctionPropagator
from dataclasses import dataclass
from autogen.trace.utils import SimplePromptParser, get_name
import autogen


"""
We follow the same design principle as trace
This file is not dependent of AutoGen library and can be used independently with trace
"""


class AbstractOptimizer:
    """An optimizer is responsible for updating the parameters based on the feedback."""

    def __init__(self, parameters: List[ParameterNode], *args, **kwargs):
        assert type(parameters) is list
        assert all([isinstance(p, ParameterNode) for p in parameters])
        self.parameters = parameters

    def step(self):
        """Update the parameters based on the feedback."""
        raise NotImplementedError

    def zero_feedback(self):
        """Reset the feedback."""
        raise NotImplementedError

    @property
    def propagator(self):
        """Return a Propagator object that can be used to propagate feedback in backward."""
        raise NotImplementedError


class Optimizer(AbstractOptimizer):
    def __init__(self, parameters: List[ParameterNode], *args, propagator: Propagator = None, **kwargs):
        super().__init__(parameters)
        propagator = propagator if propagator is not None else self.default_propagator()
        assert isinstance(propagator, Propagator)
        self._propagator = propagator

    @property
    def propagator(self):
        return self._propagator

    def step(self, *args, **kwargs):
        update_dict = self._step(self.parameters, *args, **kwargs)
        for p, d in update_dict.items():
            if p.trainable:
                p._data = d

    def zero_feedback(self):
        for p in self.parameters:
            p.zero_feedback()

    # Subclass should implement the methods below.
    def _step(self, nodes: List[ParameterNode], *args, **kwargs) -> Dict[ParameterNode, Any]:
        """Return the new data of parameter nodes based on the feedback."""
        raise NotImplementedError

    def default_propagator(self):
        """Return the default Propagator object of the optimizer."""
        raise NotImplementedError

    def backward(self, node: Node, *args, **kwargs):
        """Propagate the feedback backward."""
        return node.backward(*args, propagator=self.propagator, **kwargs)


# class DummyOptimizer(Optimizer):
#     # FOR TESTING PURPOSES ONLY
#     def _step(self, node: ParameterNode):
#         value, feedback = node.data, node.feedback
#         if isinstance(value, dict):
#             base = value["content"]
#         elif isinstance(value, str):
#             base = value
#         else:
#             raise NotImplementedError
#         new = base + " ".join([" ".join(v) for v in feedback.values()])
#         return new


class FunctionOptimizer(Optimizer):
    problem_template = dedent(
        """
        #Code
        {code}

        #Documentation
        {documentation}

        #Variables
        {variables}

        #Outputs
        {outputs}

        #Others
        {others}

        #Feedback:
        {feedback}
        """
    )

    system_message_template = dedent(
        """
        You're tasked debug and solve a coding/algorithm problem. You will see the code, the documentation of each function used in the code, and the feedback about the code's execution result. Your goal is to improve the code's output based on the feedback by changing variables used in the code.

        Specifically, a problem will be composed of the following parts:
        - #Code: the code whose results you need to improve.
        - #Documentation: the documentation of each function used in the code.
        - #Variables: the values of the variables that you need to change.
        - #Outputs: the result of the code.
        - #Others: the values of other inputs to the code, or intermediate values created through the code.
        - #Feedback: the feedback about the code's execution result.

        In #Variables, #Outputs, and #Others, the format is:
        <type> <variable_name> = <value>
        You need to change the values of the variables in #Variables to improve the code's output in accordance to #Feedback and their data types.
        The explanation in #Documentation might be incomplete and just contain high-level description of each function. You can use the values in #Others to help infer how those functions work.

        You should write down your thought process and finally make a suggestion of the desired values of #Variables in the format below.
        If no changes are needed, include TERMINATE in #Reasoning and do not output #Suggestion.

        #Reasoning
        <Your reasoning>

        #Suggestion
        <variable_1> = <suggested_value_1>
        <variable_2> = <suggested_value_2>
        ...

        Here is an example of problem instance:

        ================================

        {example_problem}

        ================================

        Below is an ideal response for the problem above.

        ================================

        {example_response}

        ================================

        Now you see problem instance:

        {problem_instance}

        """
    )

    def __init__(
        self, parameters: List[ParameterNode], config_list: List, *args, propagator: Propagator = None, **kwargs
    ):
        super().__init__(parameters, *args, propagator=propagator, **kwargs)
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.example_problem = self.problem_template.format(
            code="y = add(a,b)\nz = subtract(y, c)",
            documentation="add: add two numbers\nsubtract: subtract two numbers",
            variables="(int) a = 5",
            outputs="(int) z = 1",
            others="(int) b = 1\n(int) c = 5",
            feedback="The result of the code is not as expected. The result should be 10, but the code returns 1",
        )
        self.example_response = dedent(
            """
            #Reasoning
            In this case, the desired response would be to change the value of input a to 14, as that would make the code return 10.

            #Suggestion
            a = 10
            """
        )

    def default_propagator(self):
        """Return the default Propagator object of the optimizer."""
        return FunctionPropagator()

    def _step(self, nodes: List[ParameterNode], verbose=False, *args, **kwargs) -> Dict[ParameterNode, Any]:
        assert isinstance(self.propagator, FunctionPropagator)

        # Aggregate feedback from all the nodes
        feedbacks = [self.propagator.aggregate(node.feedback) for node in nodes]
        summary = sum(feedbacks[1:], feedbacks[0])

        # Construct variables and update others
        others = {get_name(p): p.data for p in self.parameters if not p.trainable}
        others.update(summary.others)
        variables = {get_name(p): p.data for p in self.parameters if p.trainable}

        def repr_node_value(node_dict):
            return "\n".join([f"({type(v).__name__}) {k}={v}" for k, v in node_dict.items()])

        # Format prompt
        problem_instance = self.problem_template.format(
            code="\n".join([v for k, v in sorted(summary.graph)]),
            documentation="\n".join([v for v in summary.documentation.values()]),
            variables=repr_node_value(variables),
            outputs=repr_node_value(summary.output),
            others=repr_node_value(summary.others),
            feedback=summary.user_feedback,
        )

        prompt = self.system_message_template.format(
            example_problem=self.example_problem,
            example_response=self.example_response,
            problem_instance=problem_instance,
        )
        response = self.call_llm(prompt, verbose=verbose)

        if "TERMINATE" in response:
            return {}

        # Extract the suggestion from the response
        suggestion = self.extract_suggestion(response)

        # Convert the suggestion in text into the right data type
        update_dict = {}
        for node in nodes:
            if node.trainable:
                update_dict[node] = type(node.data)(suggestion[get_name(node)])
        return update_dict

    def extract_suggestion(self, response) -> dict:
        suggestion = response.split("#Suggestion")[1]
        suggestion = suggestion.split("\n")
        output = {}
        for s in suggestion:
            if "=" in s:
                k, v = s.replace("`", "").split("=")
                output[k.strip()] = v.strip()
        return output

    def call_llm(self, prompt, verbose=False):  # TODO Get this from utils?
        """Call the LLM with a prompt and return the response."""
        if verbose:
            print("Prompt\n", prompt)

        response = (
            self.llm.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )
            .choices[0]
            .message.content
        )
        if verbose:
            print("LLM response:\n", response)
        return response


class TeacherLLMOptimizer(Optimizer):
    def __init__(self, parameters, config_list, task_description, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        system_message = dedent(
            """
        You are giving instructions to a student on how to accomplish a task.
        The student aims to get a high score.
        Given the feedback the student has received and the instruction you have given,
        You want to come up with a new instruction that will help the student get a higher score.
        """
        )
        self.llm = AssistantAgent(
            name="assistant", system_message=system_message, llm_config={"config_list": config_list}
        )
        self.task_description = task_description

        self.parameter_copy = {}
        self.save_parameter_copy()

    def _step(self, nodes: ParameterNode):
        # `prompt="Complete the following sentence: {prefix}, context={"prefix": "Today I feel"}`
        # context = {"task_description": self.task_description,
        #            "instruction": value,
        #            "feedback": feedback}
        raise NotImplementedError  # TODO

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


class LLMCallable(object):
    def __init__(self, config_list, system_message, prompt_template, name="helper"):
        self.config_list = config_list
        self.llm = AssistantAgent(name=name, system_message=system_message, llm_config={"config_list": config_list})
        self.parser = SimplePromptParser(prompt_template, verbose=False)

    def create_simple_response(self, message):
        messages = [{"content": message, "role": "user"}]
        response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
        return self.llm.client.extract_text_or_completion_object(response)[0]

    def create_response(self, **kwargs):
        results = self.parser(**kwargs)
        messages = [{"content": results, "role": "user"}]
        response = self.llm.client.create(messages=self.llm._oai_system_message + messages)
        return self.llm.client.extract_text_or_completion_object(response)[0]


if __name__ == "__main__":
    # add a few unit tests here
    pass
