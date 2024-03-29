import argparse

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
                        default=r"F:\code\myProjects\dlkt\lab\saved_models\2024-03-28@21-24-43@@qDKT_CORE@@seed_0@@our_setting_new@@assist2009_train_fold_0")
    parser.add_argument("--save_model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置
    parser.add_argument("--setting_name", type=str, default="our_setting_new")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--data_type", type=str, default="only_question",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--test_file_name", type=str, help="文件名", default="assist2009_test_fold_0.txt")
    parser.add_argument("--base_type", type=str, default="concept", choices=("concept", "question"),
                        help="如果是multi concept数据集训练，并且想使用由PYKT提出的基于习题的测试，请设置为question，其它情况都为concept")
    parser.add_argument("--evaluate_batch_size", type=int, default=256)

    # ---------------------------- 细粒度配置（不适用于base_type为question的evaluate）----------------------------------------
    # 长尾问题（注意不同训练集的长尾统计信息不一样）
    parser.add_argument("--statics_file_path", type=str, help="prepare4fine_trained_evaluate.py生成的文件绝对路径",
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0_statics.json")
    # 冷启动问题
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seq_len_absolute", type=str, help="[0, 10, 200]表示测试模型对位于序列0~10区间的点的性能以及10~200区间点的性能",
                        default="[0, 10, 100, 200]")
    # 偏差问题（习题偏差和学生偏差，测试模型对于正确率高（低）的序列中高（低）正确率习题的预测能力），需要配合statics_file_path使用
    parser.add_argument("--previous_seq_len4bias", type=int, default=5)
    parser.add_argument("--seq_most_accuracy4bias", type=float, default=0.4)

    # 习题偏差问题（论文：Do We Fully Understand Students’ Knowledge States? Identifying and Mitigating Answer Bias in Knowledge Tracing提出）
    # 该测试无需配置
    # -------------------------------------------------------------------------------------------------------------------

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

    global_objects["models"] = {}
    global_objects["data_loaders"] = {}
    model = load_kt_model(global_params, global_objects,
                          params["save_model_dir"], params["save_model_name"], params["model_name_in_ckt"])
    global_objects["models"]["kt_model"] = model
    global_objects["data_loaders"]["test_loader"] = dataloader_test

    evaluator = Evaluator(global_params, global_objects)
    if params["base_type"] == "question":
        evaluator.evaluate_base_question4multi_concept()
    else:
        evaluator.evaluate()
