import dspy
from datasets import load_dataset
from dspy.evaluate import Evaluate

import re
from tqdm import tqdm

def eval_metric(true, prediction, trace=None):
    pred = prediction.answer
    matches = re.findall(r"\([A-Z]\)", pred)
    parsed_answer = matches[-1] if matches else ""
    return parsed_answer == true.answer

class BasicQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.prog(question=question)

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

        rewards += correctness
    return rewards / len(examples), responses

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tracking_shuffled_objects_seven_objects")
    parser.add_argument("--train", action="store_true", default="Enabled few-shot optimization over training samples")
    parser.add_argument("--cot", action="store_true", default="Use and train CoT model instead")
    parser.add_argument("--save_path", type=str, default="results/bigbench_dspy")
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
    assert args.task in tasks, f"Task {args.task} not found in tasks."
    ds = load_dataset("maveriq/bigbenchhard", args.task)["train"]
    examples = [dspy.Example({"question": r["input"], "answer": r["target"]}).with_inputs("question") for r in ds]

    print(f"There are {len(examples)} examples.")
    trainset = examples[:20]
    valset = examples[20:]

    stats = {}

    llm = dspy.OpenAI(model="gpt-4-turbo-2024-04-09", max_tokens=512)
    dspy.settings.configure(lm=llm)

    basic_qa = BasicQA()
    evaluate = Evaluate(devset=valset, metric=eval_metric, num_threads=6, display_progress=True, display_table=10,
                        return_outputs=True)
    val_acc, return_outputs = evaluate(basic_qa)

    stats['train_acc'] = 0 # rewards / len(trainset)

    stats['val_acc'] = val_acc
    stats['val_responses'] = return_outputs

    import pickle
    with open(f"{args.save_path}/{args.task}.pkl", "wb") as f:
        pickle.dump(stats, f)
