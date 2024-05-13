import autogen
from autogen.trace.nodes import node, GRAPH
import string
import random
import numpy as np
from textwrap import dedent
from autogen.trace.optimizers import FunctionOptimizerV2
from datasets import load_dataset

from typing import List
import copy
from autogen.trace.trace_ops import FunModule, trace_op, trace_class, TraceExecutionError
from autogen.trace.nodes import Node

import re
from tqdm import tqdm
import ray # for parallelization

def eval_metric(true, prediction):
    # two types of answers:
    # (A)/(B) or "syndrome therefrom"/8/No/invalid
    matches = re.findall(r"\([A-Z]\)", true)
    if matches:
        pred = prediction
        matches = re.findall(r"\([A-Z]\)", pred)
        parsed_answer = matches[-1] if matches else ""
        return parsed_answer == true
    else:
        # substring match
        return prediction == true

class LLMCallable:
    def __init__(self, config_list=None, max_tokens=512, verbose=False):
        if config_list is None:
            config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.max_tokens = max_tokens
        self.verbose = verbose

    @trace_op(catch_execution_error=False)
    def call_llm(self, user_prompt):
        """
        Sends the constructed prompt (along with specified request) to an LLM.
        """
        system_prompt = "You are a helpful assistant.\n"
        if self.verbose not in (False, "output"):
            print("Prompt\n", system_prompt + user_prompt)

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        try:
            response = self.llm.create(
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.llm.create(messages=messages, max_tokens=self.max_tokens)
        response = response.choices[0].message.content

        if self.verbose:
            print("LLM response:\n", response)
        return response


@trace_class
class Predict(LLMCallable):
    def __init__(self):
        super().__init__()

        self.demos = []
        self.prompt_template = dedent("""
        Given the fields `question`, produce the fields `answer`.

        ---

        Follow the following format.

        Question: {{question}}
        Answer: {{answer}}

        ---
        Question: {}
        Answer:
        """)

        self.prompt_template = node(self.prompt_template, trainable=True,
                                    constraint="The prompt template needs to include {question} for the LLM to answer.")

    @trace_op(trainable=True)
    def extract_answer(self, response):
        answer = response.split("\nAnswer:")[1].strip()
        return answer

    @trace_op(trainable=True)
    def create_prompt(self, prompt_template, question):
        return prompt_template.format(question)

    def forward(self, question):
        """
        question: text

        We read in a question and produces a resposne
        """
        user_prompt = self.create_prompt(self.prompt_template, question)
        response = self.call_llm(user_prompt)
        return self.extract_answer(response)

def learn_predict(dp, optimizer, examples):
    # optimizer.objective = optimizer.default_objective

    cum_reward = 0
    for example in tqdm(examples):
        GRAPH.clear()
        max_calls = 2
        while max_calls > 0:
            # This is also online optimization
            # we have the opportunity to keep changing the function with each round of interaction
            try:
                response = dp.forward(example['question'])
                correctness = eval_metric(example['answer'], response.data)
                feedback = "The answer is correct! No need to change anything." if correctness else f"The answer is wrong. Please choose from the given options. The correct answer is \"{example['answer']}\". Please modify the prompt and relevant parts of the program to help LLM produce the right answer."
            except TraceExecutionError as e:
                # this is essentially a retry
                response = e.exception_node
                feedback = response.data
                correctness = False

            print(example['question'])
            print("Expected answer:", example['answer'])
            print("Answer:", response.data)

            cum_reward += correctness

            # if we can handle the case, no need to optimize
            if correctness:
                break

            optimizer.zero_feedback()
            optimizer.backward(response, feedback)

            print(f"output={response.data}, feedback={feedback}, variables=\n")  # logging
            for p in optimizer.parameters:
                print(p.name, p.data)
            optimizer.step(verbose=False)
            max_calls -= 1

    print(f"Total reward: {cum_reward}")
    return cum_reward

def evaluate_dp(dp, examples):
    rewards = 0
    responses = []
    for example in tqdm(examples):
        try:
            response = dp.forward(example['question'])
            responses.append(response.data)
            correctness = eval_metric(example['answer'], response.data)
        except:
            correctness = False
            responses.append(None)

        rewards += correctness
    return rewards / len(examples), responses

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tracking_shuffled_objects_seven_objects")
    parser.add_argument("--task_start", type=int, default=-1, help="Start from a specific task")
    parser.add_argument("--task_end", type=int, default=-1, help="End at a specific task")
    parser.add_argument("--train", action="store_true", help="We add modules to add few-shot examples")
    parser.add_argument("--cot", action="store_true", help="Use and train CoT model instead")
    parser.add_argument("--save_path", type=str, default="results/bigbench")
    args = parser.parse_args()

    import os
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tasks = ['tracking_shuffled_objects_seven_objects', 'salient_translation_error_detection',
             'tracking_shuffled_objects_three_objects', 'geometric_shapes', 'object_counting', 'word_sorting',
             'logical_deduction_five_objects', 'hyperbaton', 'sports_understanding', 'logical_deduction_seven_objects',
             'multistep_arithmetic_two', 'ruin_names', 'causal_judgement', 'logical_deduction_three_objects',
             'formal_fallacies', 'snarks', 'boolean_expressions', 'reasoning_about_colored_objects', 'dyck_languages',
             'navigate', 'disambiguation_qa', 'temporal_sequences', 'web_of_lies',
             'tracking_shuffled_objects_five_objects', 'penguins_in_a_table', 'movie_recommendation',
             'date_understanding']

    rerun_tasks = ['object_counting', 'word_sorting', 'sports_understanding', 'multistep_arithmetic_two', 'causal_judgement', 'formal_fallacies',
                    'boolean_expressions', 'dyck_languages', 'navigate', 'web_of_lies']

    assert args.task in tasks, f"Task {args.task} not found in tasks."
    # note 0:27 covers all tasks
    run_tasks = tasks[args.task_start:args.task_end] if args.task_start != -1 and args.task_end != -1 else [args.task]

    for task in run_tasks:

        print(f"Running task {task}")

        save_name = f""
        if args.train:
            save_name += "trained_"
        if args.cot:
            save_name += "cot_"
        save_name += f"{task}.pkl"

        if os.path.exists(f"{args.save_path}/{save_name}") and task not in rerun_tasks:
            print(f"Task {task} already finished and not in rerun task. Skipping.")
            continue

        train = load_dataset("maveriq/bigbenchhard", task)["train"]
        examples = [{"question": r["input"], "answer": r["target"]} for r in train]

        print(f"There are {len(examples)} examples.")
        trainset = examples[:20]
        valset = examples[20:]

        stats = {}

        dp = Predict()
        optimizer = FunctionOptimizerV2(dp.parameters() + [dp.prompt_template],
                                        config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))
        rewards = learn_predict(dp, optimizer, trainset)
        stats["learned_prompt"] = dp.prompt_template.data
        stats["extract_answer"] = dp.parameters_dict()['extract_answer'].data
        stats["create_prompt"] = dp.parameters_dict()['create_prompt'].data
        stats['optimizer_log'] = optimizer.log
        stats['train_acc'] = rewards / len(trainset)

        val_acc, responses = evaluate_dp(dp, valset)
        stats['val_acc'] = val_acc
        stats['val_responses'] = responses

        import pickle

        with open(f"{args.save_path}/{save_name}", "wb") as f:
            pickle.dump(stats, f)
