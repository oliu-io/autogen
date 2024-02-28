import copy
from autogen.trace import node
from autogen.trace import operators as ops
from autogen.trace.propagators import sum_propagate

x = node(1)
def sum_of_integers():
    y = x.clone()
    z = ops.add(x,y)
    y_clone = y.clone()
    return ops.add(z, y_clone), z

final, z = sum_of_integers()
fig = final.backward('feedback', visualize=True)
fig.view()

for k, v in x._feedback.items():
    print(f'child {k}: {k.name}: {k.data}: {v}')
assert ' '.join([str(k) for k in x._feedback.values()]) == "['feedback'] ['feedbackfeedback']"


# z.backward('z_feedback', visualize=True)
# This would throw an error because z has been backwarded.
x.zero_feedback()
final, z = sum_of_integers()
fig = final.backward('feedback', retain_graph=True, visualize=True)
fig.view()
fig = z.backward('__extra__', visualize=True)
# fig.view()  # This visualizes only the subgraph before z
print('\n')
for k, v in x._feedback.items():
    print(f'child {k}: {k.name}: {k.data}: {v}')
