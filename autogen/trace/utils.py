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

import re

# Currently do not support nested if-statement or for-loop
# Handlebar syntax: https://github.com/guidance-ai/guidance/tree/main#template-syntax
# https://handlebarsjs.com/

class SimplePromptParser:

    def __init__(self, template_text, verbose=False, reduce_linebreaks=True):
        self.verbose = verbose
        self.template_text = template_text
        self.reduce_linebreaks = reduce_linebreaks

    def __call__(self, **kwargs):
        template_text = self.template_text
        labeled_blocks = self.extract_blocks(template_text)
        if labeled_blocks[-1][0] == "assistant":
            labeled_blocks = labeled_blocks[:-1] # we remove the last assistant block, because that's for generation

        typed_messages = []
        # messages = [{"role": "system", "content": self.system_prompt},
        #             {"role": "user", "content": prompt}]
        # assert labeled_blocks[0][0] == 'system', "The first block must be a system block"

        for block_type, content in labeled_blocks:
            # if statement is handled first, because this decides if the content should stay or disappear
            content = content.strip()
            content = self.parse_if_block(content, **kwargs)
            each_keys = self.identify_loop_keywords(content)
            for key in each_keys:
                content = self.populate_template_for_each(content, key, **kwargs)
            # content = self.populate_template_for_each(content , **kwargs)
            content = self.populate_vars(content, **kwargs)
            if self.reduce_linebreaks:
                # match multiple line breaks and replace with a single line break
                content = re.sub(r"(\n\s*){2,}", "\n\n", content)
            typed_messages.append({"role": block_type, "content": content})

            if self.verbose:
                print("------New block------")
                print(f"Block Type: {block_type.upper()}")
                print(content)
                print("------End block------")

        return typed_messages

    def decode_typed_messages(self, typed_messages):
        messages = ""
        for typed_message in typed_messages:
            messages += typed_message['role'] + ': ' + typed_message['content']
        return messages

    def parse_if_block(self, parsed_text, **kwargs):
        # Regular expression to capture the content inside the {{#if ...}} and {{/if}} tags
        pattern = r"{{#if (\w+)}}(.*?){{/if}}"

        matches = re.findall(pattern, parsed_text, re.DOTALL)
        for condition_var, block_content in matches:
            # If the condition variable exists in kwargs and its value is True
            if condition_var not in kwargs:
                raise Exception(f"Key '{condition_var}' for if statement not found in provided arguments.")
            cond = kwargs[condition_var]
            if cond:
                # Remove the {{#if}} and {{/if}} tags, but keep the content
                parsed_text = parsed_text.replace(r"{{#if " + condition_var + "}}" + block_content + "{{/if}}",
                                                  block_content)
            else:
                # Remove the entire block
                parsed_text = parsed_text.replace(r"{{#if " + condition_var + "}}" + block_content + "{{/if}}", "")

        # Return the modified text
        return parsed_text

    def populate_vars(self, template, **kwargs):
        # Regular expression to find all placeholders
        placeholders = re.findall(r"{{(.*?)}}", template)

        # Replace each placeholder with its corresponding value from kwargs
        for placeholder in placeholders:
            if placeholder in kwargs:
                kwargs[placeholder] = self.none_to_empty_string(kwargs[placeholder])
                template = template.replace(f"{{{{{placeholder}}}}}", kwargs[placeholder])
            else:
                template = template.replace(f"{{{{{placeholder}}}}}", f"Placeholder {placeholder} not provided")
                raise Exception(template)

        return template

    def none_to_empty_string(self, value):
        # this is only applicable to populate_vars and for_each
        # if_exists takes None dddd
        if value is None:
            return ""
        return value

    def identify_loop_keywords(self, template):
        pattern = r"{{#each (\w+)}}"

        # Use findall to extract all matches
        keywords = re.findall(pattern, template)

        return keywords

    def populate_template_for_each(self, template, each_key, **kwargs):
        # We don't support nested for-loop

        before_each_match = re.search(r"(.*?){{#each " + each_key + "}}", template, re.DOTALL)
        before_each = before_each_match.group(1) if before_each_match else None

        # Extract the portion between {{~/each}} and {{~/user}}
        after_each_match = re.search(r"{{~/each}}(.*?)$", template, re.DOTALL)
        after_each = after_each_match.group(1)if after_each_match else None

        if each_key not in kwargs:
            return template

        examples = kwargs[each_key]

        # Regular expression to extract keys after 'this.'
        keys = re.findall(r"{{this\.(.*?)}}", template)

        # Getting the template part inside the {{~#each}} and {{~/each}} tags
        template_inside_each = re.search(r"{{#each "+each_key+"}}(.*?){{~/each}}", template, re.DOTALL).group(1)

        # Generating the text for each dictionary in examples
        populated_texts = []
        for example in examples:
            populated_text = template_inside_each
            for key in keys:
                if key in example:
                    example[key] = self.none_to_empty_string(example[key])
                    populated_text = populated_text.replace("{{this."+key+"}}", example[key])
            populated_texts.append(populated_text)

        if before_each is not None:
            populated_texts = [before_each] + populated_texts

        if after_each is not None:
            populated_texts = populated_texts + [after_each]

        return "\n".join(populated_texts)

    def extract_blocks(self, parsed_text):
        # Define regex patterns for each block type
        patterns = {
            "system": re.compile(r"{{#system~}}(.*?){{~/system}}", re.DOTALL),
            "user": re.compile(r"{{#user~}}(.*?){{~/user}}", re.DOTALL),
            "assistant": re.compile(r"{{#assistant~}}(.*?){{~/assistant}}", re.DOTALL)
        }

        # Find all occurrences of each block and label them
        labeled_blocks = []
        for block_type, pattern in patterns.items():
            for match in pattern.findall(parsed_text):
                labeled_blocks.append((block_type, match))

        # Sort by their appearance order in the parsed text
        labeled_blocks.sort(key=lambda x: parsed_text.index(x[1]))

        # if there is no block marking, then we treat it as a giant user block instead
        if len(labeled_blocks) == 0:
            labeled_blocks.append(("user", parsed_text))

        return labeled_blocks


def usage_test_1():
    # Test
    parsed_text = """
    {{#system~}}
    You are a helpful assistant that wants to come up with instructions to a student to help them write a poem that is satisfactory to a teacher's assignment.
    The student's poem needs to satisfy the requirement of this assignment.
    {{~/system}}

    {{#user~}}

    Now, you are given a new assignment, and you want to see if you can update the instructions to help the student write a poem that satisfies the new assignment.

    {{#if exists_instruction}}
    In addition, here are some helpful advice and guidance:
    {{instruction}}
    {{/if}}

    Your Instruction:
    {{~/user}}
    """

    kwargs = {
        "exists_instruction": False,
        "instruction": "Try to use metaphors and similes to add depth to your poem."
    }

    parser = SimplePromptParser(parsed_text, verbose=True)
    results = parser(**kwargs)

    print(results)


def usage_test_2():
    parsed_text = """
    {{#system~}}
    You are a helpful assistant that wants to come up with instructions to a student to help them write a poem that is satisfactory to a teacher's assignment.
    The student's poem needs to satisfy the requirement of this assignment.
    {{~/system}}

    {{#user~}}
    Here are some instructions you wrote for the previous assignments:
    {{#each examples}}
    {{role}}'s Assignment: {{this.assignment}}

    Your Instruction: 
    {{this.instruction}}
    ---------------
    {{~/each}}

    {{#each feedbacks}}
    {{role}}'s feedback: {{this.feedback}}
    ---------------
    {{~/each}}
    {{~/user}}

    {{#user~}}

    Now, you are given a new assignment, and you want to see if you can update the instructions to help the student write a poem that satisfies the new assignment.
    Teacher's Assignment: {{new_assignment}}

    Your Instruction:
    {{~/user}}
    """
    examples = [
        {"assignment": "Write about a rainy day.", "instruction": "Imagine the sound of raindrops..."},
        {"assignment": "Describe a sunny day.", "instruction": "Think of the warmth of the sun..."}
    ]

    feedbacks = [
        {"feedback": "Good job!"}
    ]

    new_assignment = "Compose a poem about winter."
    parser = SimplePromptParser(parsed_text, verbose=True)
    results = parser(examples=examples, feedbacks=feedbacks, new_assignment=new_assignment, role="Teacher")
    print(results)

    results = parser(examples=[], feedbacks=[], new_assignment=new_assignment)
    print(results)