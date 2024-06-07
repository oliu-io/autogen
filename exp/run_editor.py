import json
import autogen
from autogen.trace.nodes import node, GRAPH, ParameterNode
from textwrap import dedent
from autogen.trace.optimizers import FunctionOptimizerV2
from datasets import load_dataset

from autogen.trace.bundle import FunModule, bundle, trace_class, TraceExecutionError
from autogen.trace.nodes import Node

import re
from tqdm import tqdm

def eval_metric(instruction, reply, edit, llm):
    
    system_prompt = "You are a helpful assistant.\n"
    user_prompt = dedent(
        """
    You are given a user `instruction` and two assistant `replies`. Your goal is to compare the two replies and judge which one is better, and explain your reasoning.

    User Instruction: {}

    Reply A: {}

    Reply B: {}

    You should provide a judgment and reasoning in a JSON format. The judgment should be either "A" or "B". The reasoning should be a string explaining why you chose that reply. For example:

    {{
        "judgment": "A",
        "reasoning": "[REASONING]"
    }}
    """
    )
    messages = [
        {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt.format(instruction, reply, edit)}
    ]
    response = llm.create(
        messages=messages,
        response_format={"type": "json_object"},
    ).choices[0].message.content
    try:
        response = json.loads(response)
        judgment = response["judgment"]
        reasoning = response["reasoning"]
        assert judgment in ["A", "B"]
        is_edit_better = judgment == "B"        
    except Exception as e:
        import pdb; pdb.set_trace()
        is_edit_better = False
        reasoning = ''

    return is_edit_better, reasoning



class LLMCallable:
    def __init__(self, config_list=None, max_tokens=1024, verbose=False):
        if config_list is None:
            config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.max_tokens = max_tokens
        self.verbose = verbose

    @bundle(catch_execution_error=False)
    def call_llm(self, user_prompt):
        """
        Sends the constructed prompt (along with specified request) to an LLM.
        """
        system_prompt = "You are a helpful assistant.\n"
        if self.verbose not in (False, "output"):
            print("Prompt\n", system_prompt + user_prompt)

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

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
class Edit(LLMCallable):
    def __init__(self):
        super().__init__()

        self.demos = []
        self.prompt_template = dedent(
            """
        You are given a user `instruction` and an assistant `reply`.

        Here is some edit suggestion: Enhance the quality of your responses by including specific details, examples, and actionable advice. When providing tips or guidelines, aim to give concrete suggestions that are easy to understand and implement. Ensure that your responses are well-rounded and cover different aspects of the topic where applicable.

        You must generate a new reply that is better than the original reply, and put it in `edit`. Even if the original reply is good, you should still try to improve it.
        ---

        Follow the following format.

        Instruction: 
        Reply: 
        Edit:

        ---
        Instruction: {}
        Reply: {}
        Edit:
        """
        )

        self.prompt_template = ParameterNode(
            self.prompt_template, 
            trainable=True,
            description=(
                "[ParameterNode] This is the Edit Template to the LLM. " + \
                "Need more detailed edit suggestions to improve the edited reply. " + \
                "Need to include information about what the format of answers LLM should output. " + \
                "They can be a string, a number like 8, (A)/(B), or Yes/No."
            )
        )

    @bundle(trainable=False, catch_execution_error=True, allow_external_dependencies=True)
    def extract_answer(self, prompt_template, instruction, edit, response):
        """
        Need to read in the response, which can contain additional thought, delibration and an edited reply.
        Use code to process the response and find where the answer is.
        Can use self.call_llm("Return the answer from this text: " + response) again to refine the answer if necessary.

        Args:
            prompt_template: The prompt that was used to query LLM to get the response
            instruction: Instruction has a text describing the user instruction
            reply: Reply has a text describing the assistant's reply
            response: LLM returned a string response
                      Process it and return the edited reply in the exact format that the evaluator wants to see.
                      Be mindful of the type of reply you need to produce.
                      It can be string, a number like 8, (A)/(B), or Yes/No.
        """
        answer = response.split("Edit:")[1].strip()
        return answer

    @bundle(trainable=False, catch_execution_error=True, allow_external_dependencies=True)
    def create_prompt(self, prompt_template, instruction, reply):
        """
        The function takes in an instruction and a reply. It then add to the prompt for LLM to answer.
        Args:
            prompt_template: some guidance/hints/suggestions for LLM
            instruction: the instruction for the LLM to follow
            reply: the reply for the LLM to edit
        """
        return prompt_template.format(instruction, reply)

    def forward(self, instruction, reply):
        """
        instruction: text
        reply: text

        We read in a user instruction and an assistant reply. We then provide an edit to the reply to make it better.
        """
        user_prompt = self.create_prompt(self.prompt_template, instruction, reply)
        response = self.call_llm(user_prompt)
        answer = self.extract_answer(self.prompt_template, instruction, reply, response)
        return answer



def learn_predict(dp, optimizer, examples, val_examples, task_name, save_dir):
    # optimizer.objective = "Be mindful of the type of answer you need to produce." + optimizer.default_objective
    cum_reward = 0
    epochs = 1

    val_perfs = {}
    for epoch in range(epochs):
        for step, example in enumerate(tqdm(examples)):
            GRAPH.clear()
            # This is also online optimization
            # we have the opportunity to keep changing the function with each round of interaction
            try:
                response = dp.forward(example['instruction'], example['reply'])
                is_edit_better, reasoning = eval_metric(example['instruction'], example['reply'], response.data, dp.llm)
                if is_edit_better:
                    feedback = "The edited reply is better than the original reply. No need to change anything."
                else:
                    feedback = dedent(
                        f"""
                        The response is worse than the original reply. Here is the reasoning:
                        {reasoning}
                        """
                    )
                no_error = True
            except TraceExecutionError as e:
                # load in the previous best checkpoint, and try to optimize from that again
                # an error recovery mode (similar to MCTS!?)
                import pdb; pdb.set_trace()
                if len(val_perfs) > 0:
                    best_checkpoint = max(val_perfs, key=val_perfs.get)
                    dp.load(best_checkpoint)
                    try:
                        response = dp.forward(example['instruction'], example['reply'])
                        is_edit_better, reasoning = eval_metric(example['instruction'], example['reply'], response.data, dp.llm)
                        if is_edit_better:
                            feedback = "The edited reply is better than the original reply. No need to change anything."
                        else:
                            feedback = dedent(
                                f"""
                                The response is worse than the original reply. Here is the reasoning:
                                {reasoning}
                                """
                            )
                        no_error = True
                    except:
                        response = e.exception_node
                        feedback = response.data
                        is_edit_better = False
                        no_error = False
                else:
                    response = e.exception_node
                    feedback = response.data
                    is_edit_better = False
                    no_error = False

            print(example["instruction"])
            print("Original Reply:", example["reply"])
            print("Edited Reply:", response.data)

            cum_reward += is_edit_better
            checkpoint_name = f"{save_dir}/{task_name}/epoch_{epoch}_step_{step}.pkl"

            # if we can handle the case, no need to optimize
            if is_edit_better:
                # evaluate on val examples
                try:
                    val_perf, _ = evaluate_dp(dp, val_examples)
                    val_perfs[checkpoint_name] = val_perf
                    dp.save(checkpoint_name)
                except:
                    pass

                continue

            # if val_perf is completely empty and there is no immediate error, we save two checkpoints
            if no_error and len(val_perfs) < 2:
                try:
                    val_perf, _ = evaluate_dp(dp, val_examples)
                    val_perfs[checkpoint_name] = val_perf
                    dp.save(checkpoint_name)
                except:
                    pass

            optimizer.zero_feedback()
            optimizer.backward(response, feedback)

            print(f"output={response.data}, feedback={feedback}, variables=\n")  # logging
            for p in optimizer.parameters:
                print(p.name, p.data)
            optimizer.step(verbose=False)

    # in the end, we select the best checkpoint on validation set
    # by here we have at least one checkpoint
    best_checkpoint = max(val_perfs, key=val_perfs.get)
    print(f"Best checkpoint: {best_checkpoint}", f"Val performance: {val_perfs[best_checkpoint]}")
    dp.load(best_checkpoint)

    checkpoint_name = f"{save_dir}/{task_name}/best_ckpt.pkl"
    dp.save(checkpoint_name)

    print(f"Total reward: {cum_reward}")
    return dp, cum_reward



def evaluate_dp(dp, examples):
    rewards = 0
    responses = []
    for example in tqdm(examples):
        try:
            response = dp.forward(example["instruction"], example["reply"])
            responses.append(response.data)
            is_edit_better = eval_metric(example["instruction"], example["reply"], response.data, dp.llm)[0]
        except:
            is_edit_better = False
            responses.append(None)

        rewards += is_edit_better
    return rewards / len(examples), responses


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tracking_shuffled_objects_seven_objects")
    parser.add_argument("--task_start", type=int, default=-1, help="Start from a specific task")
    parser.add_argument("--task_end", type=int, default=-1, help="End at a specific task")
    parser.add_argument("--save_path", type=str, default="results/bigbench")
    parser.add_argument("--load_ckpt", type=str, default="")
    args = parser.parse_args()

    import os

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    task = 'tatsu-lab/alpaca'
    
    print(f"Running task {task}")

    save_name = f""
    ckpt_save_name = f"editor_{task}"

    save_name += f"{task}.pkl"

    # if os.path.exists(f"{args.save_path}/{save_name}") and task not in rerun_tasks:
    #     print(f"Task {task} already finished and not in rerun task. Skipping.")
    #     continue

    train = load_dataset(task, split='train[:200]')
    examples = [{"instruction": r['instruction'], "reply": r["output"]} for r in train]

    print(f"There are {len(examples)} examples.")
    trainset = examples[:15]
    valset = examples[15:20]  # last 5 to validate the performance
    test_set = examples[20:]

    stats = {}

    if args.load_ckpt != "" and task == args.task:
        dp = Edit()
        dp.load(args.load_ckpt)
    else:
        dp = Edit()

        optimizer = FunctionOptimizerV2(dp.parameters() + [dp.prompt_template],
                                        config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))
        dp, rewards = learn_predict(dp, optimizer, trainset, valset, task, ckpt_save_name)
        stats['optimizer_log'] = optimizer.log
        stats['train_acc'] = rewards / len(trainset)

    stats["learned_prompt"] = dp.prompt_template.data
    stats["extract_answer"] = dp.parameters_dict()['extract_answer'].data
    stats["create_prompt"] = dp.parameters_dict()['create_prompt'].data

    print(stats["extract_answer"])

    val_acc, responses = evaluate_dp(dp, test_set)
    stats['val_acc'] = val_acc
    stats['val_responses'] = responses
    print(f"Validation accuracy: {val_acc}")

    import pickle

    with open(f"{args.save_path}/{save_name}", "wb") as f:
        pickle.dump(stats, f)
