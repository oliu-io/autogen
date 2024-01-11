from graphviz import Digraph

def back_prop_node_visualization(start_node):
    dot = Digraph()
    node = start_node

    visited = set()
    stack = [start_node]

    # we do two loops because I worry Digraph requires "pre-registration" of all nodes
    # add node names
    while stack:
        current_node = stack.pop()
        print(f'Node {node.name}: Node Type {node}, Node: {node._data}')
        if current_node not in visited:
            dot.node(node.name.replace(":", ""), node.name.replace(":", ""))
            visited.add(current_node)
            stack.extend(current_node.parents)

    # add node edges
    visited = set()
    stack = [start_node]

    while stack:
        current_node = stack.pop()
        if current_node not in visited:
            for parent in current_node.parents:
                dot.edge(current_node.name.replace(":", ""), parent.name.replace(":", ""))
            visited.add(current_node)
            stack.extend(current_node.parents)

    return dot