from graphviz import Digraph


def for_all_methods(decorator):
    """ Applying a decorator to all methods of a class. """
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)) and not attr.startswith("__"):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def back_prop_node_visualization(start_node, reverse=False):
    dot = Digraph()
    node = start_node

    visited = set()
    stack = [start_node]

    # we do two loops because I worry Digraph requires "pre-registration" of all nodes
    # add node names
    while stack:
        current_node = stack.pop()
        # print(f'Node {node.name}: Node Type {node}, Node: {node._data}')
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
                if not reverse:
                    dot.edge(current_node.name.replace(":", ""), parent.name.replace(":", ""))
                else:
                    dot.edge(parent.name.replace(":", ""), current_node.name.replace(":", ""))
            visited.add(current_node)
            stack.extend(current_node.parents)

    return dot

def backfill_lists(parent_list):
    max_length = max(len(child) for child in parent_list)

    for child in parent_list:
        # While the child list is shorter than the longest, append its last element
        while len(child) < max_length:
            child.append(child[-1])

    return parent_list

def plot_agent_performance(performances, backfilled=True):
    import matplotlib.pyplot as plt
    import numpy as np

    if not backfilled:
        performances = backfill_lists(performances)

    performances = np.array(performances)

    # Calculate mean and standard deviation
    means = np.mean(performances, axis=0)
    stds = np.std(performances, axis=0)

    # Epochs
    epochs = np.arange(1, len(means) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, means, label='Mean Performance')
    plt.fill_between(epochs, means - stds, means + stds, alpha=0.2)

    # Labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.title('Performance Across Epochs with Confidence Interval')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

def verbalize(next_obs, feedback, reward):
    message = f"""Score: {reward}\n\n"""
    message += f"Feedback: {feedback}\n\n"
    if next_obs is not None:
        message += f"Instruction: {next_obs}\n\n"
    return message