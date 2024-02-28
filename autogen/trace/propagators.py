

from typing import Any
from autogen.trace.nodes import Node, MessageNode, get_operator_type, IDENTITY_OPERATORS
from collections import defaultdict


class AbtractPropagator:
    def __call__(self, child: MessageNode):
        """ Calling this method would propagte the feedback from the child to the parents."""
        assert isinstance(child, MessageNode)
        assert all([len(f)<=1 for f in child.feedback.values()]) # All MessageNode feedback should be at most length 1
        propagated_feedback = self.propagate(child.data, child.description, child.feedback, child.parents)
        # Check propagated feedback has the right format
        # It should be a dictionary with the parents as keys and the feedback as values
        assert isinstance(propagated_feedback, dict)
        assert all((p in propagated_feedback for p in child.parents))
        return propagated_feedback

    def propagate(self, data: Any, description: str, feedback: dict, parents: Node):
        """ Compute propagated feedback to node.parents based on
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

    def propagate(self, data: Any, description: str, feedback: dict, parents: Node):
        operator_type = get_operator_type(description)
        if operator_type in self.override:
            return self.override[operator_type](data, description, feedback, parents)
        else:
            return self._propagate(data, description, feedback, parents)

    def _propagate(self, data: Any, description: str, feedback: dict, parents: Node):
        """ Compute propagated feedback to node.parents based on
            node.description, node.data, and node.feedback. Return a dict where
            the keys are the parents and the values are the
            propagated feedback.
        """
        raise NotImplementedError


get_name = lambda x: x.name.replace(":", "")
def get_label(x, print_limit=200):
    if isinstance(x, str):
        return x

    text = get_name(x)+'\n'+x.description+'\n'
    content = str(x.data['content'] if isinstance(x.data, dict) else x.data)
    if len(content) > print_limit:
        content = content[:print_limit] + '...'
    return text + content


class sum_propagate(Propagator):
    def _propagate(self, data: Any, description: str, feedback: dict, parents: Node):
        # Simply sum the feedback
        feedback_list = [v[0] for k, v in feedback.items()]
        if len(feedback_list) == 0:
            summary = ''
        else:
            assert all([type(feedback_list[0]) == type(f) for f in feedback_list])
            if isinstance(feedback_list[0], str):
                summary = ''.join(feedback_list)
            else: # isinstance(feedback_list[0], int):
                summary = sum(feedback_list)
        return {parent: summary for parent in parents}

class retain_last_only_propagate(Propagator):

    def _propagate(self, data: Any, description: str, feedback: dict, parents: Node):
        summary = ''.join([v[0] for k, v in feedback.items()])
        return {parent: summary for parent in parents}

class retain_full_history_propagate(Propagator):

    def _propagate(cls, data: Any, description: str, feedback: dict, parents: Node):
        # this retains the full history
        summary = ''.join([f'\n\n{get_label(k).capitalize()}{v[0]}' for k, v in feedback.items()])
        return {parent: summary for parent in parents}
