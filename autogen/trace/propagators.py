from typing import Any, List, Dict
from autogen.trace.nodes import Node, MessageNode, get_operator_name
from collections import defaultdict
from autogen.trace.utils import SimplePromptParser, get_name
from textwrap import dedent


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
        """Compute propagated feedback to node.parents based on
        node.description, node.data, and node.feedback. Return a dict where
        the keys are the parents and the values are the
        propagated feedback.
        """
        raise NotImplementedError

    def summarize(self, node: Node) -> str:
        """Compute a text summary of the feedback for the node."""
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
            aggregated_feedback = self.aggregate(child.feedback)
            graph = graph + aggregated_feedback["graph"]
            data.update(aggregated_feedback["data"])
            doc.update(aggregated_feedback["doc"])
            user_feedback = aggregated_feedback["user_feedback"]

        summary = dict(data=data, graph=graph, doc=doc, user_feedback=user_feedback)
        return {parent: summary for parent in child.parents}

    @staticmethod
    def update_graph(graph: List[str], sub_graph: List[str]):
        for g in sub_graph:
            if g not in graph:
                graph.append(g)
        return graph

    def aggregate(self, feedback: dict):
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


class retain_last_only_propagate(Propagator):
    def _propagate(self, child: MessageNode):
        summary = "".join([v[0] for k, v in child.feedback.items()])
        return {parent: summary for parent in child.parents}


class retain_full_history_propagate(Propagator):
    def _propagate(self, child: MessageNode):
        # this retains the full history
        summary = "".join(
            [
                f"\n\n{get_label(child.description).capitalize()}:{child.data}\n\n{get_label(k).capitalize()}{v[0]}"
                for k, v in child.feedback.items()
            ]
        )
        return {parent: summary for parent in child.parents}


class self_graph_propagate(Propagator):
    """
    This propagator does these things:
    1. construct a partial graph at each node
    2. Following visualization principle, ignore identity functions

    We have the following information we want to propagate:
    1. Myopic graph structure -- only input/output of self, not the full chain
    2. Input/output of this node
    3. Delta X (how much did the input/output change)
    4. Feedback from before
    """

    def __init__(self):
        super().__init__()
        self.parser = SimplePromptParser()

        self.partial_graph_format = dedent(
            """
        Function: {{input_llms}} -> {{name}}
        {{#each inputs}}
        <Input{{this.num}}>
        Name: {{this.llm}}:
            {{this.input}}
        </Input{{this.num}}>

        {{~/each}}
        <Output>
            {{output}}
        </Output>
        """
        )

    def _propagate(self, child: MessageNode):
        # this retains the full history
        summary = "".join([f"\n\n{get_label(k).capitalize()}{v[0]}" for k, v in child.feedback.items()])
        return {parent: summary for parent in child.parents}


class partial_graph_propagate(Propagator):
    """
    This propagator does these things:
    1. construct a partial graph at each node
    2. Following visualization principle, ignore identity functions

    We have the following information we want to propagate:
    1. Graph structure (partial) (what is this node's computation afffecting output)
      - The entire chain gets rendered
      (Need parsing)
      - Since there is no loop,
    2. Input/output of this node
    3. Delta X (how much did the input/output change)
    4. Feedback from before
    """

    pass
