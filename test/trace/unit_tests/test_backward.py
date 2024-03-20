import copy
from autogen.trace import node
from autogen.trace import operators as ops

x = node(1)


def sum_of_integers():
    y = x.clone()
    z = ops.add(x, y)
    y_clone = y.clone()
    return ops.add(z, y_clone), z


final, z = sum_of_integers()
final.backward("feedback")

for k, v in x._feedback.items():
    print(f"child {k}: {k.name}: {k.data}: {v}")
assert " ".join([str(k) for k in x._feedback.values()]) == "['feedback'] ['feedbackfeedback']"
print("\n")


try:
    z.backward("z_feedback")
except Exception as e:
    print("This would throw an error because z has been backwarded.")
    print(type(e), e)
    print("\n")

x.zero_feedback()
final, z = sum_of_integers()
fig = final.backward("feedback", retain_graph=True, visualize=True, simple_visualization=False)
fig.view()
fig = z.backward("__extra__", visualize=True, simple_visualization=False)
# fig.view()  # This visualizes only the subgraph before z
for k, v in x._feedback.items():
    print(f"child {k}: {k.name}: {k.data}: {v}")
