from typing import Any, List
from autogen.trace.nodes import Node, MessageNode, get_operator_name
from collections import defaultdict
from autogen.trace.utils import SimplePromptParser
from textwrap import dedent


class AbtractPropagator:
    def __call__(self, child: MessageNode):
        """Calling this method would propagte the feedback from the child to the parents."""
        assert isinstance(child, MessageNode)
        assert all(
            [len(f) <= 1 for f in child.feedback.values()]
        )  # All MessageNode feedback should be at most length 1
        propagated_feedback = self.propagate(child.data, child.description, child.feedback, child.parents)
        # Check propagated feedback has the right format
        # It should be a dictionary with the parents as keys and the feedback as values
        assert isinstance(propagated_feedback, dict)
        assert all((p in propagated_feedback for p in child.parents))

        return propagated_feedback

    def propagate(self, data: Any, description: str, feedback: dict, parents: List[Node]):
        """Compute propagated feedback to node.parents based on
        node.description, node.data, and node.feedback. Return a dict where
        the keys are the parents and the values are the
        propagated feedback.
        """
        raise NotImplementedError


class Propagator(AbtractPropagator):
    def __init__(self):
        self.override = dict()  # key: operator name: value: override propagate function

    def register(self, operator_name, propagate_function):
        self.override[operator_name] = propagate_function

    def propagate(self, data: Any, description: str, feedback: dict, parents: List[Node]):
        operator_name = get_operator_name(description)
        if operator_name in self.override:
            return self.override[operator_name](data, description, feedback, parents)
        else:
            return self._propagate(data, description, feedback, parents)

    def _propagate(self,data: Any, description: str, feedback: dict, parents: List[Node]):
        """Compute propagated feedback to node.parents based on
        node.description, node.data, and node.feedback. Return a dict where
        the keys are the parents and the values are the
        propagated feedback.
        """
        raise NotImplementedError


def get_name(x):
    return x.name.replace(":", "")


def get_label(x, print_limit=200):
    if isinstance(x, str):
        return x

    text = get_name(x) + "\n" + x.description + "\n"
    content = str(x.data["content"] if isinstance(x.data, dict) else x.data)
    if len(content) > print_limit:
        content = content[:print_limit] + "..."
    return text + content

# Note:
# if len(feedback) > 1, it means there are two or more child nodes from this node,
# we might need to perform a "merge" feedback action

class sum_propagate(Propagator):
    def _propagate(self, data: Any, description: str, feedback: dict, parents: List[Node]):
        # Simply sum the feedback
        feedback_list = [v[0] for k, v in feedback.items()]
        if len(feedback_list) == 0:
            summary = ""
        else:
            assert all([type(feedback_list[0]) == type(f) for f in feedback_list]), "error in propagate"
            if isinstance(feedback_list[0], str):
                summary = "".join(feedback_list)
            else:  # isinstance(feedback_list[0], int):
                summary = sum(feedback_list)
        return {parent: summary for parent in parents}


class retain_last_only_propagate(Propagator):
    def _propagate(self, data: Any, description: str, feedback: dict, parents: List[Node]):
        summary = "".join([v[0] for k, v in feedback.items()])
        return {parent: summary for parent in parents}


class retain_full_history_propagate(Propagator):
    def _propagate(cls, data: Any, description: str, feedback: dict, parents: List[Node]):
        # this retains the full history
        summary = "".join([f"\n\n{get_label(description).capitalize()}:{data}\n\n{get_label(k).capitalize()}{v[0]}" for k, v in feedback.items()])
        return {parent: summary for parent in parents}


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

        self.partial_graph_format = dedent("""
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
        """)

    def _propagate(cls, data: Any, description: str, feedback: dict, parents: List[Node]):
        # this retains the full history
        summary = "".join([f"\n\n{get_label(k).capitalize()}{v[0]}" for k, v in feedback.items()])
        return {parent: summary for parent in parents}

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