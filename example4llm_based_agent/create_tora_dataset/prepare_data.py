"""
This code is copied from: https://github.com/microsoft/ToRA
"""
import argparse
import json
from parse import *

from utils import prepare_data, construct_prompt, show_sample
from PythonExecutor import PythonExecutor
from api import llm_api, api_with_func_call
from eval import math_equal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 数据位置参数
    parser.add_argument("--data_dir", type=str,
                        default="/Users/dream/Desktop/code/projects/dlkt/lab/math_dataset/dataset_raw")
    parser.add_argument("--data_name", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_dir", type=str,
                        default="/Users/dream/Desktop/code/projects/dlkt/prompt_template/tora")
    parser.add_argument("--output_dir", type=str,
                        default="/Users/dream/Desktop/code/projects/dlkt/lab/math_dataset/tora")

    # tora参数
    parser.add_argument("--prompt_type", type=str, default="tora")
    parser.add_argument("--num_test_sample", type=int, default=10, help="-1 for full data")
    parser.add_argument("--use_train_prompt_format", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # api参数
    parser.add_argument("--model_name_or_path", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_tokens_per_call", type=int, default=1024)

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

    for example in examples:
        idx = example['idx']
        example['question'] = parse_question(example, params["data_name"])
        # gt_cot是数据集中的推理过程，gt_ans是从gt_cot中提取的答案，即标签
        gt_cot, gt_ans = parse_ground_truth(example, params["data_name"])
        full_prompt = construct_prompt(params, example)

        # results的element是chatgpt的输出（不包含prompt），内容是python程序（可能有）和用box包含的答案，数量为params["n_sampling"]
        if "tora" in params["prompt_type"]:
            results = api_with_func_call(
                model_name=params["model_name_or_path"],
                prompt=full_prompt,
                max_tokens=params["max_tokens_per_call"],
                temperature=params["temperature"],
                n=params["n_sampling"],
                top_p=params["top_p"],
                executor=executor,
            )
        else:
            stop_tokens = ["</s>", "---", "```output"]
            if args.prompt_type in ['cot']:
                stop_tokens.append("\n\n")
            # use your own API like OpenAI API
            results = llm_api(
                model_name=params["model_name_or_path"],
                prompt=full_prompt,
                max_tokens=params["max_tokens_per_call"],
                temperature=params["temperature"],
                n=params["n_sampling"],
                top_p=params["top_p"],
                stop=stop_tokens,
            )

        # deal with error
        if results == ['error']:
            print(">>> Error API call")
            continue

        # get prediction
        predictions = []
        reports = []
        for r in results:
            pred, report = run_execute(executor, r, params["prompt_type"], execute=True)
            predictions.append(pred)
            reports.append(report)

        scores = [math_equal(p, gt_ans, timeout=True) for p in predictions]
        if scores[0]:
            correct += 1
        else:
            wrong += 1

        sample = {
            'idx': idx,
            'question': example['question'],
            'gt_cot': gt_cot,
            'gt': gt_ans,
            'pred': predictions,
            'score': scores
        }

        if params["prompt_type"] == "cot":
            sample.update({'code': results})
        elif "tora" in params["prompt_type"] or "pal" in params["prompt_type"]:
            sample.update({'report': reports, 'code': results})

        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]

        show_sample(sample)
        if correct + wrong > 0:
            print(f"Average accuracy of chatgpt currently: {correct / (correct + wrong)}")
        print("==" * 20)

        try:
            writer.write(json.dumps(sample) + '\n')
            writer.flush()
        except:
            print(">>> Error writing to file")
            continue

    writer.close()
