import copy
from autogen.trace import node
from autogen.trace import operators as ops


# Sum of str
x = node('NodeX')
y = node('NodeY')
z = ops.add(x=x, y=y)
print(f'Sum of Node[str]')
print(f" x:{x.data}\n y:{y.data}\n z:{z.data}")

assert z.data == x.data + y.data
assert x in z.parents and y in z.parents
assert z in x.children and z in y.children
for k,v in z._inputs.items():
    assert locals()[k] == v

# Sum of intergers
x = node(1)
y = node(2)
z = ops.add(x,y)
print(f'Sum of Node[int]')
print(f" x:{x.data}\n y:{y.data}\n z:{z.data}")
assert z.data == x.data + y.data
assert x in z.parents and y in z.parents
assert z in x.children and z in y.children
for k,v in z._inputs.items():
    assert locals()[k] == v

# Condition
condition = node(True)
z = ops.cond(condition, x, y)
assert z.data == x.data if condition.data else y.data
assert x in z.parents and y in z.parents and condition in z.parents
assert z in x.children and z in y.children and z in condition.children
for k,v in z._inputs.items():
    assert locals()[k] == v

# Getitem of list of Nodes
index = node(0)
x = node([node(1), node(2), node(3)])
z = ops.getitem(x, index)
assert z.data == x.data[index.data].data
assert x in z.parents and index in z.parents
assert z in x.children and z in index.children
for k,v in z._inputs.items():
    assert locals()[k] == v

# Getitem of list
index = node(0)
x = node([1,2,3])
z = ops.getitem(x, index)
assert z.data == x.data[index.data]
assert x in z.parents and index in z.parents
assert z in x.children and z in index.children
for k,v in z._inputs.items():
    assert locals()[k] == v

# Test copy
z_new = ops.identity(z)
z_clone = z.clone()
z_copy = copy.deepcopy(z)

assert z_new.data == z.data
assert z_clone.data == z.data
assert z_copy.data == z.data
assert z in z_new.parents and len(z_new.parents) == 1 and z_new in z.children
assert z in z_clone.parents and len(z_clone.parents) == 1 and z_clone in z.children
assert z not in z_copy.parents and len(z_copy.parents) == 0 and z_copy not in z.children


# Test magic function
x = node('NodeX')
y = node('NodeY')
z = x+y
print(f'Sum of Node[str]')
print(f" x:{x.data}\n y:{y.data}\n z:{z.data}")

assert z.data == x.data + y.data
assert x in z.parents and y in z.parents
assert z in x.children and z in y.children
for k,v in z._inputs.items():
    assert locals()[k] == v