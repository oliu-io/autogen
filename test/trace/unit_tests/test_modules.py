from autogen.trace.modules import to_data, Module
from autogen.trace.nodes import Node, node
from autogen.trace.utils import for_all_methods
from autogen.trace.trace_ops import trace_op, trace_class

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

# Test Module as a class

class TestClass(Module):
    @trace_op(trainable=True)
    def method1(self, x):
        return x

    def method2(self, y):
        return y

@trace_class
class TestClass2:
    @trace_op(trainable=True)
    def method1(self, x):
        return x

    def method2(self, y):
        return y
def test_parameters():
    t = TestClass()
    assert len(t.parameters()) == 1
    assert len(t.parameters_dict()) == 1

test_parameters()

def test_parameters2():
    t = TestClass2()
    assert len(t.parameters()) == 1
    assert len(t.parameters_dict()) == 1

test_parameters2()