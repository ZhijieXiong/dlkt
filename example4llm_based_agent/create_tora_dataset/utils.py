import os
import json
import random
from datetime import datetime


KEY_MAP = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def load_jsonl(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


def prepare_data(params):
    data_file_path = os.path.join(params["data_dir"], params["data_name"], f"{params['split']}.jsonl")
    assert os.path.exists(data_file_path), f"{data_file_path} not exist"
    examples = list(load_jsonl(data_file_path))
    for i, example in enumerate(examples):
        example["idx"] = i

    if params["num_test_sample"] > 0:
        examples = random.sample(examples, params["num_test_sample"])
    elif params["num_test_sample"] < 0:
        params["num_test_sample"] = len(examples)
    else:
        raise "num_test_sample must not be 0"

    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(params["model_name_or_path"].split("/")[-2:])
    out_file_prefix = f'{params["split"]}_{params["prompt_type"]}_{params["num_test_sample"]}_seed{params["seed"]}_t{params["temperature"]}'
    out_file_path = os.path.join(params["output_dir"], model_name, params["data_name"], f"{out_file_prefix}_{dt_string}.jsonl")
    os.makedirs(f'{params["output_dir"]}/{model_name}/{params["data_name"]}', exist_ok=True)

    # load all processed samples
    processed_files = [
        f
        for f in os.listdir(f"{params['output_dir']}/{model_name}/{params['data_name']}/")
        if f.endswith(".jsonl") and f.startswith(out_file_prefix)
    ]
    processed_samples = []
    for f in processed_files:
        processed_samples.extend(list(load_jsonl(f"{params['output_dir']}/{model_name}/{params['data_name']}/{f}")))

    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_indices = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example['idx'] not in processed_indices]

    return examples, processed_samples, out_file_path


def load_prompt(prompt_dir, data_name, prompt_type):
    assert data_name in ["gsm8k", "math"], f"Dataset {data_name} is not supported at the moment"
    if prompt_type in ['platypus_fs', 'wizard_zs']:
        prompt_type = "cot"
    prompt_path = os.path.join(prompt_dir, prompt_type, f"{data_name}.md")
    if not os.path.exists(prompt_path):
        prompt_path = "./prompts/{}.md".format(prompt_type)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as fp:
            prompt = fp.read().strip() + "\n\n"
    else:
        print(f"Error: prompt file {prompt_path} not found")
        prompt = ""
    return prompt


def construct_prompt(params, example):
    demo_prompt = load_prompt(params["prompt_dir"], params["data_name"], params["prompt_type"])
    if params["use_train_prompt_format"]:
        full_prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
    elif "tora" in params["prompt_type"]:
        context = f"Question: {example['question']}\n\nSolution:"
        full_prompt = demo_prompt + context
    elif params["prompt_type"] in ["direct", "cot"]:
        context = f"Question: {example['question']}\nAnswer:"
        full_prompt = demo_prompt + context
    elif params["prompt_type"] == "pal":
        context = f"Question: {example['question']}"
        full_prompt = demo_prompt + context
    elif params["prompt_type"] == "wizard_zs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif params["prompt_type"] == "platypus_fs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt.format(instruction=demo_prompt + f"Question: {example['question']}\nAnswer:")
    else:
        raise NotImplementedError(params["prompt_type"])

    return full_prompt


def extract_program(result: str, last_only=True):
    """
    extract the program after "```python", and before "```"
    """
    program = ""
    start = False
    for line in result.split("\n"):
        if line.startswith("```python"):
            if last_only:
                # only extract the last program
                program = ""
            else:
                program += "\n# ========\n"
            start = True
        elif line.startswith("```"):
            start = False
        elif start:
            program += line + "\n"
    return program


def show_sample(sample, print_all_predict=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample['question']))
    if 'code' in sample:
        if print_all_predict:
            for code in sample['code']:
                print('-' * 20)
                print("code:", code)
            print("Execution:", sample['report'])
        else:
            print("Solution:\n", sample['code'][0])
            print("Execution:", sample['report'][0])
    if 'pred' in sample:
        print("Prediction:", repr(sample['pred'][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = KEY_MAP.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
