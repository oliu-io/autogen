from dataclasses import dataclass
from typing import Any, List, Dict, Tuple
from autogen.trace.nodes import Node, MessageNode, get_operator_name
from collections import defaultdict
from autogen.trace.utils import SimplePromptParser, get_name
from textwrap import dedent
import autogen


class AbtractPropagator:
    def __call__(self, child: MessageNode):
        """Calling this method would propagte the feedback from the child to the parents."""
        assert isinstance(child, MessageNode)
        assert all(
            [len(f) <= 1 for f in child.feedback.values()]
        )  # All MessageNode feedback should be at most length 1
        # TODO maybe just pass node
        propagated_feedback = self.propagate(child)
        # Check propagated feedback has the right format
        # It should be a dictionary with the parents as keys and the feedback as values
        assert isinstance(propagated_feedback, dict)
        assert all((p in propagated_feedback for p in child.parents))
        return propagated_feedback

    def propagate(self, child: MessageNode) -> Dict[Node, Any]:
        """Compute propagated feedback to node.parents of a node. Return a dict where
        the keys are the parents and the values are the
        propagated feedback.
        """
        raise NotImplementedError


class Propagator(AbtractPropagator):
    def __init__(self):
        self.override = dict()  # key: operator name: data: override propagate function

    def register(self, operator_name, propagate_function):
        self.override[operator_name] = propagate_function

    def propagate(self, child: MessageNode) -> Dict[Node, Any]:
        operator_name = get_operator_name(child.description)
        if operator_name in self.override:
            return self.override[operator_name](child)
        else:
            return self._propagate(child)

    def _propagate(self, child: MessageNode) -> Dict[Node, Any]:
        """Compute propagated feedback to node.parents based on
        node.description, node.data, and node.feedback. Return a dict where
        the keys are the parents and the values are the
        propagated feedback.
        """
        raise NotImplementedError


# Note:
# if len(feedback) > 1, it means there are two or more child nodes from this node,
# we might need to perform a "merge" feedback action


class SumPropagator(Propagator):
    # TODO remove this or add prompt
    def _propagate(self, child: MessageNode):
        # Simply sum the feedback
        feedback_list = [v[0] for k, v in child.feedback.items()]
        if len(feedback_list) == 0:
            summary = ""
        else:
            assert all([type(feedback_list[0]) == type(f) for f in feedback_list]), "error in propagate"
            if isinstance(feedback_list[0], str):
                summary = "".join(feedback_list)
            else:  # isinstance(feedback_list[0], int):
                summary = sum(feedback_list)
        return {parent: summary for parent in child.parents}


@dataclass
class FunctionFeedback:
    """Feedback container used by FunctionPropagator."""

    graph: List[
        Tuple[int, str]
    ]  # Each item is is a representation of function call. The items are topologically sorted.
    documentation: Dict[str, str]  # Function name and its documentationstring
    others: Dict[str, Any]  # Intermediate variable names and their data
    roots: Dict[str, Any]  # Root variable name and its data
    _output: Dict[str, Any]  # Leaf variable name and its data
    user_feedback: str  # User feedback at the leaf of the graph

    def __init__(
        self,
        *,
        graph: List[Tuple[int, str]],
        documentation: Dict[str, str],
        others: Dict[str, Any],
        roots: Dict[str, Any],
        user_feedback: str,
        output: Dict[str, Any],
    ):
        self.others = others
        self.roots = roots
        self.graph = graph
        self.documentation = documentation
        self.user_feedback = user_feedback
        self.output = output

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        assert isinstance(value, dict) and len(value) == 1
        self._output = value

    def __add__(self, other):
        """Merge two FunctionFeedback objects and return a new FunctionFeedback object."""
        # This operator is commutative

        assert isinstance(other, FunctionFeedback)

        # Create a copy
        graph = self.graph.copy()
        others = self.others.copy()
        roots = self.roots.copy()
        documentation = self.documentation.copy()
        user_feedback = self.user_feedback
        output = self.output.copy()

        # Update the copy
        self.update_graph(graph, other.graph)
        others.update(other.others)
        roots.update(other.roots)
        documentation.update(other.documentation)
        assert self.output == other.output, "output should be the same for all children"
        assert self.user_feedback == other.user_feedback, "user feedback should be the same for all children"

        return FunctionFeedback(
            graph=graph,
            documentation=documentation,
            others=others,
            user_feedback=user_feedback,
            output=output,
            roots=roots,
        )

    @staticmethod
    def update_graph(graph: List[str], sub_graph: List[str]):
        for g in sub_graph:
            if g not in graph:
                graph.append(g)


# Distributive
class FunctionPropagator(Propagator):
    """A propagator that summarizes the graph and the feedback in terms of calls of function calls."""

    def _propagate(self, child: MessageNode):
        # Construct the function_call.
        # It should read as a function call. e.g. "output = fun_name(x,y,z)"
        function_call = self.repr_function_call(child)

        # TODO remove these. Since they're not declared and propagator is used persistently
        self.child = child
        self.function_call = function_call

        # Construct the propagated feedback.
        # The feedback from children might share subgraphs. Only add unique ones.
        graph = [(child.level, function_call)]
        documentation = {child.info["fun_name"]: child.description}  # TODO which description to use/ how to format it?

        roots = {get_name(parent): parent.data for parent in child.parents if parent.is_root}
        if "user" in child.feedback:  # This is the leaf node where the feedback is given.
            assert len(child.feedback) == 1, "user feedback should be the only feedback"
            assert len(child.feedback["user"]) == 1
            user_feedback = child.feedback["user"][0]
            feedback = FunctionFeedback(
                graph=graph,
                documentation=documentation,
                others={},  # there's no other intermediate nodes
                roots=roots,
                user_feedback=user_feedback,
                output={get_name(child): child.data},  # This node is the output, not intermediate nodes
            )

        else:  # This is an intermediate node
            aggregated_feedback = self.aggregate(child.feedback)
            feedback = aggregated_feedback + FunctionFeedback(
                graph=graph,
                documentation=documentation,
                others={get_name(child): child.data},  # record the data of the child,
                roots=roots,
                # since there should be only one
                user_feedback=aggregated_feedback.user_feedback,
                output=aggregated_feedback.output,
            )

        # post-process the graph
        feedback.graph = self._post_process_graph(feedback.graph)
        return {parent: feedback for parent in child.parents}

    @staticmethod
    def repr_function_call(child: MessageNode):
        function_call = f"{get_name(child)} = {child.info['fun_name']}("
        for parent in child.parents:
            function_call += f"{get_name(parent)}, "
        function_call = function_call[:-2] + ")"
        return function_call

    def aggregate(self, feedback: Dict[Node, List[FunctionFeedback]]):
        """Aggregate feedback from multiple children"""
        values = [v[0] for v in feedback.values()]
        return sum(values[1:], values[0])

    def _post_process_graph(self, graph: List[str]):
        return graph


class FunctionSumPropagator(FunctionPropagator):
    """
    This allows us to only reason about 2 processes
    """

    def _post_process_graph(self, graph: List[str]):
        # now we do a post-process step
        # grab the graph at the current level, if there are > 1 nodes
        # we perform a merge (early sum)
        # single chain representation
        current_level_graph = list(filter(lambda x: x[0] == self.child.level + 1, graph))
        if len(current_level_graph) > 1:
            collect_all = []
            for level, node in current_level_graph:
                func_form = node.split(" = ")[1]
                collect_all.append(func_form)
            # instead of being literal total derivative
            # we can use "Merge" as a symbol instead
            new_graph = " + ".join(collect_all)
            # graph = [(self.child.level, self.function_call), (self.child.level + 1, new_graph)]
            # resolve and flatten the level
            child_name, func_form = self.function_call.split(" = ")
            new_graph = new_graph.replace(child_name, func_form)
            graph = [(self.child.level, new_graph)]

        return graph


class FunctionDistributivePropagate(FunctionPropagator):
    def _post_process_graph(self, graph: List[str]):
        # now we do a post-process step
        # we perform a distributive sum
        current_level_graph = list(filter(lambda x: x[0] == self.child.level + 1, graph))
        if len(current_level_graph) > 1:
            collect_all = []
            for level, node in current_level_graph:
                func_form = node.split(" = ")[1]
                child_name, child_func_form = self.function_call.split(" = ")
                # we keep the path derivative
                new_graph = func_form.replace(child_name, child_func_form)
                collect_all.append((self.child.level, new_graph))
            graph = collect_all

        return graph


class LLMCallable(object):
    def __init__(self, config_list):
        autogen.OpenAIWrapper(config_list=config_list)


def test_case_shallow_diamond(prop_func):
    from autogen.trace import node

    a = node(1, name="node_c")
    b = node(2, name="node_b")
    c = a + b  # f
    d = c + 1  # g
    e = c + 2  # h
    y = d + e
    y.backward(visualize=True, feedback="Correct", propagator=prop_func())

    print(a.feedback)


if __name__ == "__main__":
    test_case_shallow_diamond(FunctionPropagator)
    test_case_shallow_diamond(FunctionSumPropagator)
    test_case_shallow_diamond(FunctionDistributivePropagate)
