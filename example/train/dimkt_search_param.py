import argparse
import os
from copy import deepcopy
from hyperopt import fmin, tpe, hp
from torch.utils.data import DataLoader

from config.dimkt_config import dimkt_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.model.DIMKT import DIMKT
from lib.trainer.KnowledgeTracingTrainer import KnowledgeTracingTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="our_setting")
    parser.add_argument("--dataset_name", type=str, default="moocradar-C_746997")
    parser.add_argument("--data_type", type=str, default="only_question",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, default="moocradar-C_746997_train_fold_0.txt")
    parser.add_argument("--valid_file_name", type=str, default="moocradar-C_746997_valid_fold_0.txt")
    parser.add_argument("--test_file_name", type=str, default="moocradar-C_746997_test_fold_0.txt")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 训练策略
    parser.add_argument("--train_strategy", type=str, default="valid_test",
                        choices=("valid_test", "no_test"))
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--epoch_early_stop", type=int, default=10)
    parser.add_argument("--use_last_average", type=str2bool, default=False)
    parser.add_argument("--epoch_last_average", type=int, default=5)
    # 评价指标选择
    parser.add_argument("--main_metric", type=str, default="AUC")
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--multi_metrics", type=str, default="[('AUC', 1), ('ACC', 1)]")
    # 学习率
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=True)
    parser.add_argument("--lr_schedule_type", type=str, default="StepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step", type=int, default=10)
    parser.add_argument("--lr_schedule_milestones", type=str, default="[5]")
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=128)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # DIMKT数据处理参数
    parser.add_argument("--num_min_question", type=int, default=15)
    parser.add_argument("--num_min_concept", type=int, default=30)
    # 模型参数
    parser.add_argument("--num_concept", type=int, default=265)
    parser.add_argument("--num_question", type=int, default=550)
    parser.add_argument("--dim_emb", type=int, default=64)
    parser.add_argument("--num_question_diff", type=int, default=100)
    parser.add_argument("--num_concept_diff", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    # sample weight
    parser.add_argument("--use_sample_reweight", type=str2bool, default=False)
    parser.add_argument("--sample_reweight_method", type=str, default="IPS-seq",
                        choices=("IPS-double", "IPS-seq", "IPS-question"))
    parser.add_argument("--IPS_min", type=float, default=0.7)
    parser.add_argument("--IPS_his_seq_len", type=int, default=20)
    # 其它
    parser.add_argument("--seed", type=int, default=0)

    def objective(parameters):
        global current_best_performance

        args = parser.parse_args()
        params = vars(args)

        # 替换参数
        params["search_params"] = True
        params["save_model"] = False
        params["debug_mode"] = False
        params["trace_epoch"] = False
        params["use_cpu"] = False
        for param_name in parameters:
            params[param_name] = parameters[param_name]

        set_seed(params["seed"])
        global_params, global_objects = dimkt_config(params)

        valid_params = deepcopy(global_params)
        valid_params["datasets_config"]["dataset_this"] = "valid"
        dataset_valid = KTDataset(valid_params, global_objects)
        dataloader_valid = DataLoader(dataset_valid, batch_size=params["evaluate_batch_size"], shuffle=False)

        train_params = deepcopy(global_params)
        train_params["datasets_config"]["dataset_this"] = "train"
        dataset_train = KTDataset(train_params, global_objects)
        dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)

        if params["train_strategy"] == "valid_test":
            test_params = deepcopy(global_params)
            test_params["datasets_config"]["dataset_this"] = "test"
            dataset_test = KTDataset(test_params, global_objects)
            dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)
        else:
            dataloader_test = None

        global_objects["data_loaders"] = {}
        global_objects["data_loaders"]["train_loader"] = dataloader_train
        global_objects["data_loaders"]["valid_loader"] = dataloader_valid
        global_objects["data_loaders"]["test_loader"] = dataloader_test

        global_objects["models"] = {}
        model = DIMKT(global_params, global_objects).to(global_params["device"])
        global_objects["models"]["kt_model"] = model
        trainer = KnowledgeTracingTrainer(global_params, global_objects)
        trainer.train()

        # DIMKT会生成difficulty文件，搜参时需要删除
        setting_dir = global_objects["file_manager"].get_setting_dir(params["setting_name"])
        train_file_name = params["train_file_name"]
        difficulty_info_path = os.path.join(setting_dir, train_file_name.replace(".txt", "_dimkt_diff.json"))
        if os.path.exists(difficulty_info_path):
            os.remove(difficulty_info_path)
        performance_this = trainer.train_record.get_evaluate_result("valid", "valid")["main_metric"]

        if performance_this > current_best_performance:
            current_best_performance = performance_this
            print("current best params:")
            print(", ".join(list(map(lambda s: f"{s}: {parameters[s]}", parameters.keys()))))
            print("current best performance:")
            print(performance_this)
        return -performance_this


    parameters_space = {
        "weight_decay": [0, 0.0001, 0.00001],
        "learning_rate": [0.001],
        "dim_emb": [64, 128],
        "num_question_diff": [25, 50, 100],
        "num_concept_diff": [25, 50, 100],
        "dropout": [0.1, 0.2],
        "num_min_question": [10],
        "num_min_concept": [10, 30]
    }
    space = {
        param_name: hp.choice(param_name, param_space)
        for param_name, param_space in parameters_space.items()
    }
    current_best_performance = 0
    fmin(objective, space, algo=tpe.suggest, max_evals=100)