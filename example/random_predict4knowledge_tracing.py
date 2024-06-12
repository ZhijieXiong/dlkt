import argparse
import os
import logging
import sys

from torch.utils.data import DataLoader

from config import FILE_MANAGER_ROOT

from lib.util.set_up import set_seed
from lib.model.RandomKTPredictor import RandomKTPredictor
from lib.dataset.KTDataset import KTDataset
from lib.evaluator.Evaluator import Evaluator
from lib.util.parse import question2concept_from_Q, concept2question_from_Q, str2bool
from lib.util.FileManager import FileManager
from lib.util.data import read_preprocessed_file


def config(local_params):
    global_params = {}
    global_objects = {}

    # 配置文件管理器和日志
    file_manager = FileManager(FILE_MANAGER_ROOT)
    global_objects["file_manager"] = file_manager
    global_objects["logger"] = logging.getLogger("evaluate_log")
    global_objects["logger"].setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    global_objects["logger"].addHandler(ch)

    # 数据集加载
    setting_name = local_params["setting_name"]
    data_type = local_params["data_type"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    test_file_name = local_params["test_file_name"]

    dataset_train_path = os.path.join(file_manager.get_setting_dir(setting_name), train_file_name)
    dataset_valid_path = os.path.join(file_manager.get_setting_dir(setting_name), valid_file_name)
    dataset_test_path = os.path.join(file_manager.get_setting_dir(setting_name), test_file_name)
    dataset_train = read_preprocessed_file(dataset_train_path)
    dataset_valid = read_preprocessed_file(dataset_valid_path)
    dataset_test = read_preprocessed_file(dataset_test_path)

    global_objects["random_kt_predictor"] = {
        "dataset_train": dataset_train,
        "dataset_valid": dataset_valid,
        "dataset_test": dataset_test
    }

    # 测试集配置
    global_params["datasets_config"] = {
        "data_type": data_type,
        "dataset_this": "test",
        "test": {
            "type": "kt",
            "setting_name": setting_name,
            "file_name": test_file_name,
            "unuseful_seq_keys": {"user_id"},
            "kt": {},
        }
    }

    # 细粒度测试配置
    train_statics_common_path = local_params["train_statics_common_path"]
    train_statics_special_path = local_params["train_statics_special_path"]
    max_seq_len = local_params["max_seq_len"]
    seq_len_absolute = local_params["seq_len_absolute"]

    global_params["evaluate"] = {"fine_grain": {}}
    fine_grain_config = global_params["evaluate"]["fine_grain"]
    fine_grain_config["max_seq_len"] = max_seq_len
    fine_grain_config["seq_len_absolute"] = eval(seq_len_absolute)
    fine_grain_config["train_statics_common_path"] = train_statics_common_path
    fine_grain_config["train_statics_special_path"] = train_statics_special_path

    # Q_table
    dataset_name = local_params["dataset_name"]
    global_objects["data"] = {}

    if data_type == "only_question":
        global_objects["data"]["Q_table"] = file_manager.get_q_table(dataset_name, "multi_concept")
    else:
        global_objects["data"]["Q_table"] = file_manager.get_q_table(dataset_name, data_type)
    Q_table = global_objects["data"]["Q_table"]
    if Q_table is not None:
        global_objects["data"]["question2concept"] = question2concept_from_Q(Q_table)
        global_objects["data"]["concept2question"] = concept2question_from_Q(Q_table)

    global_objects["data"]["Q_table_single_concept"] = file_manager.get_q_table(dataset_name, "single_concept")
    Q_table_single_concept = global_objects["data"]["Q_table_single_concept"]
    if Q_table_single_concept is not None:
        global_objects["data"]["question2concept_single_concept"] = question2concept_from_Q(Q_table_single_concept)
        global_objects["data"]["concept2question_single_concept"] = concept2question_from_Q(Q_table_single_concept)

    # 参数
    max_context_seq_len = local_params["max_context_seq_len"]
    weight_context = local_params["weight_context"]
    weight_concept = local_params["weight_concept"]

    global_params["random_kt_predictor"] = {
        "max_context_seq_len": max_context_seq_len,
        "weight_context": weight_context,
        "weight_concept": weight_concept
    }

    return global_params, global_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 测试配置
    parser.add_argument("--setting_name", type=str, default="our_setting_new")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--data_type", type=str, default="only_question",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, help="文件名", default="assist2009_train_fold_0.txt")
    parser.add_argument("--valid_file_name", type=str, help="文件名", default="assist2009_valid_fold_0.txt")
    parser.add_argument("--test_file_name", type=str, help="文件名", default="assist2009_test_fold_0.txt")

    # 参数配置
    parser.add_argument("--max_context_seq_len", type=int, default=40)
    parser.add_argument("--weight_context", type=float, default=0.4)
    parser.add_argument("--weight_concept", type=float, default=0.4)

    # 随机种子
    parser.add_argument("--seed", type=int, default=0)

    # ------------------------------------------- 细粒度指标配置 ----------------------------------------------------------
    # 是否进行细粒度测试
    parser.add_argument("--use_fine_grained_evaluation", type=str2bool, default=True)
    parser.add_argument("--train_statics_common_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0_statics_common.json")
    parser.add_argument("--train_statics_special_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0_statics_special.json")
    # 冷启动问题
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seq_len_absolute", type=str, help="[0, 10, 200]表示测试模型对位于序列0~10区间的点的性能以及10~200区间点的性能",
                        default="[0, 10, 100, 200]")
    # 偏差问题（习题偏差和学生偏差，测试模型对于正确率高（低）的序列中高（低）正确率习题的预测能力），需要配合statics_file_path使用
    # 习题偏差问题（论文：Do We Fully Understand Students’ Knowledge States? Identifying and Mitigating Answer Bias in Knowledge Tracing提出）
    # 该测试无需配置

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])

    global_params_, global_objects_ = config(params)
    global_params_["device"] = "cpu"

    model = RandomKTPredictor(global_params_, global_objects_)
    model.evaluate_valid()
    model.evaluate_test()

    if params["use_fine_grained_evaluation"]:
        dataset_test_ = KTDataset(global_params_, global_objects_)
        dataloader_test = DataLoader(dataset_test_, batch_size=64, shuffle=False)

        global_objects_["models"] = {}
        global_objects_["data_loaders"] = {}
        global_objects_["models"]["kt_model"] = model
        global_objects_["data_loaders"]["test_loader"] = dataloader_test

        evaluator = Evaluator(global_params_, global_objects_)
        evaluator.evaluate()
