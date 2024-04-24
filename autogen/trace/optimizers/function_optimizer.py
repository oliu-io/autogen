from typing import Any, List, Dict, Union, Tuple
from dataclasses import dataclass
from autogen.trace.nodes import ParameterNode, Node, MessageNode
from autogen.trace.optimizers.optimizers import Optimizer

from autogen.trace.propagators import NodeFeedback, NodePropagator
from textwrap import dedent, indent
from autogen.trace.propagators.propagators import Propagator
import autogen
import warnings
import json


def repr_function_call(child: MessageNode):
    function_call = f"{child.py_name} = {child.info['fun_name']}("
    for k, v in child.inputs.items():
        function_call += f"{k}={v.py_name}, "
    function_call = function_call[:-2] + ")"
    return function_call


def node_to_function_feedback(node_feedback: NodeFeedback):
    """Convert a NodeFeedback to a FunctionFeedback"""
    depth = 0 if len(node_feedback.graph) == 0 else node_feedback.graph[-1][0]
    graph = []
    others = {}
    roots = {}
    output = {}
    documentation = {}

    visited = set()
    for level, node in node_feedback.graph:
        # the graph is already sorted
        visited.add(node)

        if node.is_root:  # Need an or condition here
            roots.update({node.py_name: (node.data, node._constraint)})
        else:
            # Some might be root (i.e. blanket nodes) and some might be intermediate nodes
            # Blanket nodes belong to roots
            if all([p in visited for p in node.parents]):
                # this is an intermediate node
                assert isinstance(node, MessageNode)
                documentation.update({node.info["fun_name"]: node.description})
                graph.append((level, repr_function_call(node)))
                if level == depth:
                    output.update({node.py_name: (node.data, node._constraint)})
                else:
                    others.update({node.py_name: (node.data, node._constraint)})
            else:
                # this is a blanket node (classified into roots)
                roots.update({node.py_name: (node.data, node._constraint)})

    return FunctionFeedback(
        graph=graph,
        others=others,
        roots=roots,
        output=output,
        user_feedback=node_feedback.user_feedback,
        documentation=documentation,
    )


@dataclass
class FunctionFeedback:
    """Feedback container used by FunctionPropagator."""

    graph: List[
        Tuple[int, str]
    ]  # Each item is is a representation of function call. The items are topologically sorted.
    documentation: Dict[str, str]  # Function name and its documentationstring
    others: Dict[str, Any]  # Intermediate variable names and their data
    roots: Dict[str, Any]  # Root variable name and its data
    output: Dict[str, Any]  # Leaf variable name and its data
    user_feedback: str  # User feedback at the leaf of the graph


class Examples:
    def __init__(self, size: int):
        self.size = size
        self.examples = []

    def add(self, example: str):
        self.examples.append(example)
        self.examples = self.examples[-self.size :]

    def __iter__(self):
        return iter(self.examples)


class FunctionOptimizer(Optimizer):
    problem_template = dedent(
        """
        #Code
        {code}

        #Documentation
        {documentation}

        #Variables
        {variables}

        #Inputs
        {inputs}

        #Others
        {others}

        #Outputs
        {outputs}

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
        - #Inputs: the values of other inputs to the code
        - #Others: the intermediate values created through the code execution.
        - #Outputs: the result of the code.
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

    example_prompt = dedent(
        """

        Here are some feasible but not optimal solutions (i.e. not causing errors but not necessarily achieving desired behaviors) for the problem instance above:

        {examples}

        """
    )

    default_objective = (
        "Your goal is to improve the code's output based on the feedback by changing variables used in the code."
    )

    def __init__(
        self,
        parameters: List[ParameterNode],
        config_list: List = None,
        *args,
        propagator: Propagator = None,
        objective: Union[None, str] = None,
        ignore_extraction_error: bool = True,  # ignore the type conversion error when extracting updated values from LLM's suggestion
        n_feasible_solutions: bool = 0,
        **kwargs,
    ):
        super().__init__(parameters, *args, propagator=propagator, **kwargs)
        self.ignore_extraction_error = ignore_extraction_error
        if config_list is None:
            config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.objective = objective or self.default_objective
        self.example_problem = self.problem_template.format(
            code="y = add(x=a,y=b)\nz = subtract(x=y, y=c)",
            documentation="add: add x and y \nsubtract: subtract y from x",
            variables="(int) a = 5",
            outputs="(int) z = 1",
            others="(int) y = 6",
            inputs="(int) b = 1\n(int) c = 5",
            feedback="The result of the code is not as expected. The result should be 10, but the code returns 1",
        )
        self.example_response = dedent(
            """
            {"reasoning": 'In this case, the desired response would be to change the value of input a to 14, as that would make the code return 10.',
             "suggestion": {"a": 10}
            }
            """
        )

        self.feasible_solutions = Examples(n_feasible_solutions)

    def default_propagator(self):
        """Return the default Propagator object of the optimizer."""
        return NodePropagator()

    def summarize(self):
        # Aggregate feedback from all the parameters
        feedbacks = [self.propagator.aggregate(node.feedback) for node in self.parameters if node.trainable]
        summary = sum(feedbacks[1:], feedbacks[0]) if len(feedbacks) > 1 else feedbacks[0]  # NodeFeedback
        # Construct variables and update others
        # Some trainable nodes might not receive feedback, because they might not be connected to the output
        summary = node_to_function_feedback(summary)
        # Classify the root nodes into variables and others
        # summary.variables = {p.py_name: p.data for p in self.parameters if p.trainable and p.py_name in summary.roots}

        trainable_param_dict = {p.py_name: p for p in self.parameters if p.trainable}
        summary.variables = {
            py_name: data for py_name, data in summary.roots.items() if py_name in trainable_param_dict
        }
        summary.inputs = {
            py_name: data for py_name, data in summary.roots.items() if py_name not in trainable_param_dict
        }  # non-variable roots
        return summary

    def _step(self, verbose=False, *args, **kwargs) -> Dict[ParameterNode, Any]:
        assert isinstance(self.propagator, NodePropagator)
        summary = self.summarize()

        def repr_node_value(node_dict):
            temp_list = []
            for k, v in node_dict.items():
                if "__code" not in k:
                    if v[1] is not None:
                        temp_list.append(f"({type(v[0]).__name__}) {k}={v[0]} ### Allowed values: {v[1]}")
                    else:
                        temp_list.append(f"({type(v[0]).__name__}) {k}={v[0]}")
                else:
                    if v[1] is not None:
                        # In current implementation of trace_op, this case is not possible
                        temp_list.append(f"(code) {k}:{v[0]} ### Constraints: {v[1]}")
                    else:
                        temp_list.append(f"(code) {k}:{v[0]}")
            return "\n".join(temp_list)

        if not summary.user_feedback.startswith("TraceExecutionError"):  # feasible
            self.feasible_solutions.add(summary.variables)

        # Format prompt
        problem_instance = self.problem_template.format(
            code="\n".join([v for k, v in sorted(summary.graph)]),
            documentation="\n".join([v for v in summary.documentation.values()]),
            variables=repr_node_value(summary.variables),
            inputs=repr_node_value(summary.inputs),
            outputs=repr_node_value(summary.output),
            others=repr_node_value(summary.others),
            feedback=summary.user_feedback,
        )

        prompt = self.system_message_template.format(
            objective=self.objective,
            example_problem=self.example_problem,
            example_response=self.example_response,
            problem_instance=problem_instance,
        )

        if self.feasible_solutions is not None:
            examples_str = ""
            for i, example in enumerate(self.feasible_solutions):
                examples_str += f"Example {i+1}:\n{repr_node_value(example)}\n\n"
            prompt += self.example_prompt.format(examples=examples_str)

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
            suggestion = {}

        # Convert the suggestion in text into the right data type
        update_dict = {}
        for node in self.parameters:
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

    def call_llm(self, prompt: str, verbose: Union[bool, str] = False):  # TODO Get this from utils?
        """Call the LLM with a prompt and return the response."""
        if verbose not in (False, "output"):
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


# class FunctionDistributiveOptimizer(FunctionOptimizer):
#     def default_propagator(self):
#         """Return the default Propagator object of the optimizer."""
#         return FunctionDistributivePropagate()
