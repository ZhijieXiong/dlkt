import argparse
import torch

from torch.utils.data import DataLoader

from evaluate_config import evaluate_general_config

from lib.util.load_model import load_kt_model
from lib.dataset.KTDataset import KTDataset
from lib.dataset.KTDataset_cpu2device import KTDataset_cpu2device
from lib.evaluator.Evaluator import Evaluator
from lib.util.parse import str2bool
from lib.util.data import load_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device配置
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)

    # 加载模型参数配置
    parser.add_argument("--save_model_dir", type=str, help="绝对路径",
                        default=r"F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\AuxInfoDCT\baseline\2024-05-22@10-37-44@@AuxInfoDCT@@seed_0@@our_setting_new@@ednet-kt1_train_fold_0")
    parser.add_argument("--save_model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置
    parser.add_argument("--setting_name", type=str, default="our_setting_new")
    parser.add_argument("--dataset_name", type=str, default="ednet-kt1")
    parser.add_argument("--data_type", type=str, default="only_question",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--test_file_name", type=str, help="文件名", default="ednet-kt1_test_fold_0.txt")
    parser.add_argument("--base_type", type=str, default="concept", choices=("concept", "question"),
                        help="如果是multi concept数据集训练，并且想使用由PYKT提出的基于习题的测试，请设置为question，其它情况都为concept")
    parser.add_argument("--evaluate_batch_size", type=int, default=256)

    # ---------------------------- 细粒度配置（不适用于base_type为question的evaluate）---------------
    # 由prepare4fine_trained_evaluate.py生成
    parser.add_argument("--train_statics_common_path", type=str, default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\ednet-kt1_train_fold_0_statics_common.json")
    parser.add_argument("--train_statics_special_path", type=str, default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\ednet-kt1_train_fold_0_statics_special.json")
    # 冷启动问题
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seq_len_absolute", type=str, help="[0, 10, 200]表示测试模型对位于序列0~10区间的点的性能以及10~200区间点的性能",
                        default="[0, 10, 100, 200]")
    # 偏差问题（习题偏差和学生偏差，测试模型对于正确率高（低）的序列中高（低）正确率习题的预测能力）
    # 需要[train_file]_statics_common.json文件

    # 习题偏差问题（论文：Do We Fully Understand Students’ Knowledge States? Identifying and Mitigating Answer Bias in Knowledge Tracing提出）
    # 长尾问题：需要[train_file]_statics_special.json文件
    # --------------------------------------------------------------------------------------------

    # -----------------------特殊配置：如DIMKT需要统计习题难度信息-------------------------------------
    # 如果是DIMKT，需要训练集数据的difficulty信息
    parser.add_argument("--is_dimkt", type=str2bool, default=False)
    parser.add_argument("--train_diff_file_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting\edi2020-task34_train_fold_0_dimkt_diff.json")
    parser.add_argument("--num_question_diff", type=int, default=100)
    parser.add_argument("--num_concept_diff", type=int, default=100)
    # --------------------------------------------------------------------------------------------

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = evaluate_general_config(params)

    model = load_kt_model(global_params, global_objects,
                          params["save_model_dir"], params["save_model_name"], params["model_name_in_ckt"])

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
            num_concept = global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_concept"]
            diff_fuse_table = [0] * num_concept
            for c_id, c_diff_id in concept_difficulty.items():
                diff_fuse_table[c_id] = c_diff_id
            global_objects["dimkt"]["diff_fuse_table"] = torch.LongTensor(diff_fuse_table).to(global_params["device"])
            global_params["datasets_config"]["test"]["type"] = "kt4dimkt"
            global_params["datasets_config"]["test"]["kt4dimkt"] = {}
            global_params["datasets_config"]["test"]["kt4dimkt"]["num_question_difficulty"] = params["num_question_diff"]
            global_params["datasets_config"]["test"]["kt4dimkt"]["num_concept_difficulty"] = params["num_concept_diff"]
        elif params["is_dct"]:
            global_params["datasets_config"]["test"]["type"] = "agg_aux_info"
            global_params["datasets_config"]["test"]["agg_aux_info"] = {}
        dataset_test = KTDataset(global_params, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)

    global_objects["models"] = {}
    global_objects["data_loaders"] = {}
    global_objects["models"]["kt_model"] = model
    global_objects["data_loaders"]["test_loader"] = dataloader_test

    evaluator = Evaluator(global_params, global_objects)
    if params["base_type"] == "question":
        evaluator.evaluate_base_question4multi_concept()
    else:
        evaluator.evaluate()
