

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

    @classmethod
    def propagate(cls, data: Any, description: str, feedback: dict, parents: Node):
        """ Compute propagated feedback to node.parents based on
            node.description, node.data, and node.feedback. Return a dict where
            the keys are the parents and the values are the
            propagated feedback.
        """
        raise NotImplementedError


# Add a subclass that allows registering mechanism for propagators


get_name = lambda x: x.name.replace(":", "")
def get_label(x, print_limit=200):
    if isinstance(x, str):
        return x

    text = get_name(x)+'\n'+x.description+'\n'
    content = str(x.data['content'] if isinstance(x.data, dict) else x.data)
    if len(content) > print_limit:
        content = content[:print_limit] + '...'
    return text + content


class retain_last_only_propagate(AbtractPropagator):
    @classmethod
    def propagate(cls, data: Any, description: str, feedback: dict, parents: Node):
        summary = ''.join([v[0] for k, v in feedback.items()])
        return {parent: summary for parent in parents}

class retain_full_history_propagate(AbtractPropagator):
    @classmethod
    def propagate(cls, data: Any, description: str, feedback: dict, parents: Node):
        # this retains the full history
        summary = ''.join([f'\n\n{get_label(k).capitalize()}{v[0]}' for k, v in feedback.items()])
        return {parent: summary for parent in parents}
