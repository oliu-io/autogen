from autogen.trace.modules import to_data
from autogen.trace.nodes import Node, node
from autogen.trace.utils import for_all_methods

def simple_test_unnested():
    a = node(1)
    copy_a = to_data(a)

    a = node({'2': 2})
    copy_a = to_data(a)

    a = node([1, 2, 3])
    copy_a = to_data(a)


def simple_test_node_over_container():
    a = node([node(1), node(2), node(3)])
    copy_a = to_data(a)


def simple_test_container_over_node():
    a = [node(1), node(2), node(3)]
    copy_a = to_data(a)

def test_container_over_container_over_node():
    # currently fails, and we don't expect this to work
    a = ({node(1): node('1')},)
    copy_a = to_data(a)

def test_node_over_container_over_container_over_node():
    # currently fails
    a = node(({node(1): node('1')},))
    copy_a = to_data(a)

# test_container_over_container_over_node()

test_node_over_container_over_container_over_node()
simple_test_unnested()
simple_test_node_over_container()
simple_test_container_over_node()