from typing import Any, List, Dict, Union
from autogen.trace.nodes import ParameterNode, Node
from collections import defaultdict
from autogen import AssistantAgent
from autogen.oai.completion import Completion
from textwrap import dedent, indent
from copy import copy
from autogen.trace.propagators import Propagator, FunctionPropagator, FunctionDistributivePropagate
from dataclasses import dataclass
import autogen
import warnings
import json

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
        You're tasked debug and solve a coding/algorithm problem. You will see the code, the documentation of each function used in the code, and the feedback about the code's execution result.

        Specifically, a problem will be composed of the following parts:
        - #Code: the code whose results you need to improve.
        - #Documentation: the documentation of each function used in the code.
        - #Variables: the values of the variables that you need to change.
        - #Outputs: the result of the code.
        - #Others: the values of other inputs to the code, or intermediate values created through the code.
        - #Feedback: the feedback about the code's execution result.

        In #Variables, #Outputs, and #Others, the format is:
        <type> <variable_name> = <value>
        You need to change the <value> of the variables in #Variables to improve the code's output in accordance to #Feedback and their data types specified in <type>. If <type> is (code), it means <value> is the source code of a python code, which may include docstring and definitions.
        The explanation in #Documentation might be incomplete and just contain high-level description of each function. You can use the values in #Others to help infer how those functions work.

        Objective: {objective}

        Output format:

        You should write down your thought process and finally make a suggestion of the desired values of #Variables. You cannot change the lines of code in #Code but only the values in #Variables. When <type> of a variable is (code), you should write the new definition in the format of python code without syntax errors.
        Your output should be in the following json format, satisfying the json syntax (json does support single quotes):
        {{
        "reasoning": <Your reasoning>,
        "suggestion": {{
            <variable_1>: <suggested_value_1>,
            <variable_2>: <suggested_value_2>,
        }}
        }}

        If no changes are needed, just output TERMINATE_UPDATE as opposed to the json format above.


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
        self,
        parameters: List[ParameterNode],
        config_list: List,
        *args,
        propagator: Propagator = None,
        objective: Union[None, str] = None,
        ignore_extraction_error: bool = True,
        **kwargs,
    ):
        super().__init__(parameters, *args, propagator=propagator, **kwargs)
        self.ignore_extraction_error = ignore_extraction_error
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.objective = (
            objective
            or "Your goal is to improve the code's output based on the feedback by changing variables used in the code."
        )
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
            {"reasoning": 'In this case, the desired response would be to change the value of input a to 14, as that would make the code return 10.',
             "suggestion": {"a": 10}
            }
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
        others = {p.py_name: p.data for p in self.parameters if not p.trainable}
        others.update(summary.others)
        variables = {p.py_name: p.data for p in self.parameters if p.trainable}
        non_variable_roots = {k: v for k, v in summary.roots.items() if k not in variables}
        others.update(non_variable_roots)

        def repr_node_value(node_dict):
            return "\n".join(
                [
                    f"({type(v).__name__}) {k}={v}" if "__code" not in k else f"(code) {k}:{v}"
                    for k, v in node_dict.items()
                ]
            )

        # Format prompt
        problem_instance = self.problem_template.format(
            code="\n".join([v for k, v in sorted(summary.graph)]),
            documentation="\n".join([v for v in summary.documentation.values()]),
            variables=repr_node_value(variables),
            outputs=repr_node_value(summary.output),
            others=repr_node_value(others),
            feedback=summary.user_feedback,
        )

        prompt = self.system_message_template.format(
            objective=self.objective,
            example_problem=self.example_problem,
            example_response=self.example_response,
            problem_instance=problem_instance,
        )
        response = self.call_llm(prompt, verbose=verbose)

        if "TERMINATE_UPDATE" in response:
            return {}

        # Extract the suggestion from the response
        try:
            suggestion = json.loads(response)["suggestion"]
        except json.JSONDecodeError:  # TODO try to fix it
            response = response.replace("'", '"')
            print("LLM returns invalid format, cannot extract suggestions from JSON")
            print(response)

        # Convert the suggestion in text into the right data type
        update_dict = {}
        for node in nodes:
            if node.trainable and node.py_name in suggestion:
                try:
                    update_dict[node] = type(node.data)(suggestion[node.py_name])
                except (ValueError, KeyError) as e:
                    # catch error due to suggestion missing the key or wrong data type
                    if self.ignore_extraction_error:
                        warnings.warn(
                            f"Cannot convert the suggestion '{suggestion[node.py_name]}' for {node.py_name} to the right data type"
                        )
                    else:
                        raise e

        return update_dict

    def call_llm(self, prompt, verbose=False):  # TODO Get this from utils?
        """Call the LLM with a prompt and return the response."""
        if verbose:
            print("Prompt\n", prompt)

        try:  # Try tp force it to be a json object
            response = self.llm.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.llm.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        response = response.choices[0].message.content

        if verbose:
            print("LLM response:\n", response)
        return response


class FunctionDistributiveOptimizer(FunctionOptimizer):
    def default_propagator(self):
        """Return the default Propagator object of the optimizer."""
        return FunctionDistributivePropagate()


if __name__ == "__main__":
    # add a few unit tests here
    pass
