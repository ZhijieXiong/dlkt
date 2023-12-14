import argparse
import os
import torch

from torch.utils.data import DataLoader

from evaluate_config import evaluate_general_config

from lib.dataset.KTDataset import KTDataset
from lib.dataset.KTDataset_cpu2device import KTDataset_cpu2device
from lib.evaluator.Evaluator import Evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基本配置
    parser.add_argument("--save_model_dir", type=str, default=r"F:\code\myProjects\dlkt\lab\saved_models\2023-11-28-18-06-54@@AKT@@seed_0@@random_split_leave_multi_out_setting@@assist2012_train_split_5@@valid_test@@early_stop_200_10@@265-53091-64-True-2-8-256-512-False-0.2")
    parser.add_argument("--save_model_name", type=str, default="kt_model.pth")
    parser.add_argument("--setting_name", type=str, default="random_split_leave_multi_out_setting")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--test_file_name", type=str, default="assist2012_test_split_5.txt")
    parser.add_argument("--base_type", type=str, default="concept", choices=("concept", "question"))
    parser.add_argument("--dataset_name", type=str, default="assist2012",
                        help="if choose question as base_type")
    parser.add_argument("--evaluate_batch_size", type=int, default=512)

    # 细粒度配置（暂时不适用于question evaluate）
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seq_len_absolute", type=str,
                        default="[0, 5, 10, 20, 30]",
                        choices=("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                                 "[0, 5, 10, 20, 30, 40, 50, 60, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]"))
    parser.add_argument("--statics_file_name", type=str, default="assist2012_train_split_5_statics.json")

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = evaluate_general_config(params)
    if params["base_type"] == "question" and params["data_type"] == "multi_concept":
        datasets_config = global_params["datasets_config"]
        datasets_config["test"]["file_name"] = (
            datasets_config["test"]["file_name"].replace(".txt", "_question_base4multi_concept.txt"))
        dataset_test = KTDataset_cpu2device(global_params, global_objects)
    else:
        dataset_test = KTDataset(global_params, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)

    save_model_path = os.path.join(params["save_model_dir"], params["save_model_name"])
    model = torch.load(save_model_path).to(global_params["device"])
    global_objects["models"]["kt_model"] = model
    global_objects["data_loaders"]["test_loader"] = dataloader_test

    evaluator = Evaluator(global_params, global_objects)
    if params["base_type"] == "question":
        evaluator.evaluate_base_question4multi_concept()
    else:
        evaluator.evaluate()
