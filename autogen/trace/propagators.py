from typing import Any, List, Dict
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

    def summarize(self, node: Node) -> Any:
        """Compute a summary of the feedback at the node. The returned value may
        be different node.feedback and may include other information such as the
        data of the node and the description of the node."""
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

    def summarize(self, node: Node) -> Any:
        return node.feedback


def format(x):
    return x.name.replace(":", "")


def get_label(x, print_limit=200):
    if isinstance(x, str):
        return x

    text = format(x) + "\n" + x.description + "\n"
    content = str(x.data["content"] if isinstance(x.data, dict) else x.data)
    if len(content) > print_limit:
        content = content[:print_limit] + "..."
    return text + content


# Note:
# if len(feedback) > 1, it means there are two or more child nodes from this node,
# we might need to perform a "merge" feedback action


class sum_propagate(Propagator):
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


# Distributive
class function_propagate(Propagator):
    def _propagate(self, child: MessageNode):
        """Summarize the feedback in terms of code."""

        # A summary is a dictionary with the following keys
        # data: (dict) variable_name and its data
        # graph: (list of str) each element is is a representation of function call
        # doc: (dict) function_name and its docstring
        # user_feedback: (str) user feedback at the leaf of the graph

        # Construct the function_call.
        # It should read as a function call. e.g. "output = fun_name(x,y,z)"
        function_call = f"{get_name(child)} = {child.info['fun_name']}("
        for parent in child.parents:
            function_call += f"{get_name(parent)}, "
        function_call = function_call[:-2] + ")"

        self.child = child
        self.function_call = function_call

        # Only add unique ones
        graph = [(child.level, function_call)]
        data = {get_name(child): str(child.data)}  # record the data of the child
        doc = {child.info["fun_name"]: child.description}  # TODO which description to use/ how to format it?

        if "user" in child.feedback:
            assert len(child.feedback) == 1, "user feedback should be the only feedback"
            v = child.feedback["user"]
            assert len(v) == 1
            user_feedback = v[0]
        else:
            aggregated_feedback = self._aggregate(child.feedback)
            graph = graph + aggregated_feedback["graph"]
            data.update(aggregated_feedback["data"])
            doc.update(aggregated_feedback["doc"])
            user_feedback = aggregated_feedback["user_feedback"]

        graph = self._post_process(graph)

        summary = dict(data=data, graph=graph, doc=doc, user_feedback=user_feedback)
        return {parent: summary for parent in child.parents}

    @staticmethod
    def repr_function_call(child: MessageNode):
        function_call = f"{get_name(child)} = {child.info['fun_name']}("
        for parent in child.parents:
            function_call += f"{get_name(parent)}, "
        function_call = function_call[:-2] + ")"
        return function_call

    @staticmethod
    def update_graph(graph: List[str], sub_graph: List[str]):
        for g in sub_graph:
            if g not in graph:
                graph.append(g)
        return graph

    def _aggregate(self, feedback: dict):
        """Aggregate feedback from multiple children"""
        user_feedback = set()
        graph = []
        data = {}
        doc = {}
        # aggregate feedback
        for k, v in feedback.items():
            assert isinstance(v, list) and len(v) <= 1
            # Some children might not have feedback, since they're not
            # connected the node to which the feedback is provided.
            if len(v) > 0:
                v = v[0]
                assert type(v) is dict  # TODO use a dataclass instead?
                user_feedback.add(v["user_feedback"])
                self.update_graph(graph, v["graph"])
                data.update(v["data"])
                doc.update(v["doc"])
        assert len(user_feedback) == 1, "user feedback should be the same for all children"
        user_feedback = user_feedback.pop()
        return dict(data=data, graph=graph, doc=doc, user_feedback=user_feedback)

    def _post_process(self, graph: List[str]):
        return graph

    def summarize(self, node: Node) -> Any:
        summary = self._aggregate(node.feedback)
        summary["data"].update(
            {get_name(node): node.data}
        )  # Add the data of x, since summary only contains data of the children
        return summary


class function_sum_propagate(function_propagate):
    """
    This allows us to only reason about 2 processes
    """

    def _post_process(self, graph: List[str]):
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


class function_distributive_propagate(function_propagate):
    def _post_process(self, graph: List[str]):
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
        build_manager = autogen.OpenAIWrapper(config_list=config_list)


def test_case_shallow_diamond(prop_func):
    from autogen.trace import node

    a = node(1, name="node_c")
    b = node(2, name="node_b")
    c = a + b  # f
    d = c + 1  # g
    e = c + 2  # h
    y = d + e
    y.backward(visualize=True, feedback="Correct", propagate=prop_func())

    print(a.feedback)


if __name__ == '__main__':
    test_case_shallow_diamond(function_propagate)
    test_case_shallow_diamond(function_sum_propagate)
    test_case_shallow_diamond(function_distributive_propagate)
