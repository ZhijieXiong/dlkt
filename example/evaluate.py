import argparse
import os
import torch

from torch.utils.data import DataLoader

from evaluate_config import evaluate_general_config

from lib.dataset.KTDataset import KTDataset
from lib.evaluator.Evaluator import Evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基本配置
    parser.add_argument("--save_model_dir", type=str, default=r"F:\code\myProjects\dlkt\lab\saved_models\2023-11-23-05-46-03@@random_split_leave_multi_out_setting@@assist2009_train_split_6@@valid_test@@early_stop_50_10@@265-53091-64-64-128-128-gru-1-0.1-3-128-relu")
    parser.add_argument("--save_model_name", type=str, default="kt_model.pth")
    parser.add_argument("--setting_name", type=str, default="random_split_leave_multi_out_setting")
    parser.add_argument("--data_type", type=str, default="multi_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--test_file_name", type=str, default="assist2009_valid_split_6.txt")
    parser.add_argument("--base_type", type=str, default="concept", choices={"concept", "question"})
    parser.add_argument("--evaluate_batch_size", type=int, default=256)

    # 细粒度配置
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seq_len_absolute", type=str, default="[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]")
    parser.add_argument("--seq_len_percent", type=str, default="[0, .05, 1, .2, .5, .7, .9, 1]")

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = evaluate_general_config(params)
    dataset_test = KTDataset(global_params, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)

    save_model_path = os.path.join(params["save_model_dir"], params["save_model_name"])
    model = torch.load(save_model_path).to(global_params["device"])
    global_objects["models"]["kt_model"] = model
    global_objects["data_loaders"]["test_loader"] = dataloader_test

    evaluator = Evaluator(global_params, global_objects)
    evaluator.evaluate()
