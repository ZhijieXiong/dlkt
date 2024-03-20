import argparse
from tqdm import tqdm
from parse import *

from utils import prepare_data, construct_prompt
from PythonExecutor import PythonExecutor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        default="/Users/dream/Desktop/code/projects/dlkt/lab/math_dataset/dataset_raw")
    parser.add_argument("--data_name", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_dir", type=str,
                        default="/Users/dream/Desktop/code/projects/dlkt/prompt_template/tora")

    parser.add_argument("--model_name_or_path", type=str, default="gpt-4")
    parser.add_argument("--output_dir", type=str,
                        default="/Users/dream/Desktop/code/projects/dlkt/lab/math_dataset/tora")
    parser.add_argument("--prompt_type", type=str, default="tora")

    parser.add_argument("--num_test_sample", type=int, default=-1, help="-1 for full data")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_tokens_per_call", type=int, default=1024)
    parser.add_argument("--use_train_prompt_format", action="store_true")
    args = parser.parse_args()

    params = vars(args)
    examples, processed_samples, out_file_path = prepare_data(params)

    # init python executor
    if "pal" in params["prompt_type"]:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    writer = open(out_file_path, 'w')
    correct, wrong = 0, 0

    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']
        example['question'] = parse_question(example, params["data_name"])
        gt_cot, gt_ans = parse_ground_truth(example, params["data_name"])
        full_prompt = construct_prompt(params, example)

    print("")
