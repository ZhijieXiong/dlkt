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
    parser.add_argument("--save_model_dir", type=str, default=r"F:\code\myProjects\dlkt\lab\saved_models\2023-12-08-10-31-32@@DKT@@seed_0@@pykt_setting@@ednet-kt1_train_fold_0@@188-64-64-gru-1-0.3-1-256-sigmoid")
    parser.add_argument("--save_model_name", type=str, default="kt_model.pth")
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--data_type", type=str, default="multi_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--test_file_name", type=str, default="ednet-kt1_test.txt")
    parser.add_argument("--base_type", type=str, default="question", choices=("concept", "question"))
    parser.add_argument("--dataset_name", type=str, default="ednet-kt1",
                        help="if choose question as base_type")
    parser.add_argument("--evaluate_batch_size", type=int, default=512)

    # 细粒度配置
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seq_len_absolute", type=str,
                        default="[0, 5, 10, 20, 30]",
                        choices=("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                                 "[0, 5, 10, 20, 30, 40, 50, 60, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]"))
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
    if params["base_type"] == "question":
        evaluator.evaluate_base_question4multi_concept()
    else:
        evaluator.evaluate()
