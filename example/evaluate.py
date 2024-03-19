import argparse
import os
import torch

from torch.utils.data import DataLoader

from evaluate_config import evaluate_general_config

from lib.dataset.KTDataset import KTDataset
from lib.dataset.KTDataset_cpu2device import KTDataset_cpu2device
from lib.evaluator.Evaluator import Evaluator
from lib.util.parse import str2bool
from lib.util.data import load_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基本配置
    parser.add_argument("--save_model_dir", type=str, help="绝对路径",
                        default=r"F:\code\myProjects\dlkt\lab\saved_models\save\LPKT\2024-03-13@19-46-09@@LPKT@@seed_0@@our_setting@@xes3g5m_train_fold_0")
    parser.add_argument("--save_model_name", type=str, help="文件名", default="kt_model.pth")
    parser.add_argument("--setting_name", type=str, default="our_setting")
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--data_type", type=str, default="only_question",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--test_file_name", type=str, help="文件名", default="xes3g5m_test_fold_0.txt")
    parser.add_argument("--base_type", type=str, default="concept", choices=("concept", "question"))
    parser.add_argument("--evaluate_batch_size", type=int, default=256)

    # ---------------------------- 细粒度配置（不适用于base_type为question的evaluate）----------------------------------------
    # 长尾问题（注意不同训练集的长尾统计信息不一样）
    parser.add_argument("--statics_file_path", type=str, help="绝对路径",
                        default=r"")
    # 冷启动问题
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seq_len_absolute", type=str,
                        default="[0, 10, 100, 200]")
    # ---------------------------- 细粒度配置（不适用于base_type为question的evaluate）----------------------------------------

    # 是否将head question的知识迁移到zero shot question
    parser.add_argument("--transfer_head2zero", type=str2bool, default=False)
    parser.add_argument("--head2tail_transfer_method", type=str, default="mean_pool",
                        choices=("mean_pool", "max_pool", "zero_pad", "most_popular"))

    # 如果是DIMKT，需要训练集数据的difficulty信息
    parser.add_argument("--is_dimkt", type=str2bool, default=False)
    parser.add_argument("--train_diff_file_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting\edi2020-task34_train_fold_0_dimkt_diff.json")
    parser.add_argument("--num_question_diff", type=int, default=100)
    parser.add_argument("--num_concept_diff", type=int, default=100)

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = evaluate_general_config(params)
    if params["base_type"] == "question" and params["data_type"] == "multi_concept":
        dataset_test = KTDataset_cpu2device(global_params, global_objects)
    else:
        if params["is_dimkt"]:
            difficulty_info = load_json(params["train_diff_file_path"])
            question_difficulty = {}
            concept_difficulty = {}
            for k, v in difficulty_info["question_difficulty"].items():
                question_difficulty[int(k)] = v
            for k, v in difficulty_info["concept_difficulty"].items():
                concept_difficulty[int(k)] = v
            global_objects["dimkt"] = {}
            global_objects["dimkt"]["question_difficulty"] = question_difficulty
            global_objects["dimkt"]["concept_difficulty"] = concept_difficulty
            global_params["datasets_config"]["test"]["type"] = "kt4dimkt"
            global_params["datasets_config"]["test"]["kt4dimkt"] = {}
            global_params["datasets_config"]["test"]["kt4dimkt"]["num_question_difficulty"] = params["num_question_diff"]
            global_params["datasets_config"]["test"]["kt4dimkt"]["num_concept_difficulty"] = params["num_concept_diff"]
        dataset_test = KTDataset(global_params, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)
    save_model_path = os.path.join(params["save_model_dir"], params["save_model_name"])
    model = torch.load(save_model_path).to(global_params["device"])

    global_objects["models"] = {}
    global_objects["data_loaders"] = {}
    global_objects["models"]["kt_model"] = model
    global_objects["data_loaders"]["test_loader"] = dataloader_test
    evaluator = Evaluator(global_params, global_objects)
    if params["base_type"] == "question":
        evaluator.evaluate_base_question4multi_concept()
    else:
        evaluator.evaluate()
