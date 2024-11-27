import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from config.akt_config import akt_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.model.AKT import AKT
from lib.trainer.KnowledgeTracingTrainer import KnowledgeTracingTrainer


# 如果要使用IPS-double或者IPS-question这两种样本损失重加权方法
# 请先运行prepare4fine_trained_evaluate.py获得数据集中习题的common statics信息
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="our_setting")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, default="statics2011_train_fold_0.txt")
    parser.add_argument("--valid_file_name", type=str, default="statics2011_valid_fold_0.txt")
    parser.add_argument("--test_file_name", type=str, default="statics2011_test_fold_0.txt")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 训练策略
    parser.add_argument("--train_strategy", type=str, default="valid_test",
                        choices=("valid_test", "no_test"))
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--epoch_early_stop", type=int, default=10)
    parser.add_argument("--use_last_average", type=str2bool, default=False, help="目前未实现")
    parser.add_argument("--epoch_last_average", type=int, default=5, help="目前未实现")
    # 评价指标选择
    parser.add_argument("--main_metric", type=str, default="AUC", choices=("AUC", "ACC", "RMSE", "MAE"))
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--multi_metrics", type=str, default="[('AUC', 1), ('ACC', 1)]")
    # 学习率
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=False)
    parser.add_argument("--lr_schedule_type", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step", type=int, default=10)
    parser.add_argument("--lr_schedule_milestones", type=str, default="[5, 10]")
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--evaluate_batch_size", type=int, default=24)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 模型参数
    parser.add_argument("--num_concept", type=int, default=27)
    parser.add_argument("--num_question", type=int, default=1223)
    parser.add_argument("--dim_model", type=int, default=64)
    parser.add_argument("--key_query_same", type=str2bool, default=True)
    parser.add_argument("--num_head", type=int, default=4)
    parser.add_argument("--num_block", type=int, default=1)
    parser.add_argument("--dim_ff", type=int, default=64)
    parser.add_argument("--dim_final_fc", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--separate_qa", type=str2bool, default=False)
    parser.add_argument("--seq_representation", type=str, default="encoder_output",
                        help="主要用于对比学习，在CL4KT中使用的是knowledge_encoder_output作为序列的表征，"
                             "但是也可以选择encoder_output作为序列表征，实测区别不大",
                        choices=("encoder_output", "knowledge_encoder_output"))
    parser.add_argument("--weight_rasch_loss", type=float, default=0.00001)
    # sample weight
    parser.add_argument("--use_sample_reweight", type=str2bool, default=False)
    parser.add_argument("--sample_reweight_method", type=str, default="IPS-seq",
                        choices=("IPS-double", "IPS-seq", "IPS-question"))
    parser.add_argument("--IPS_min", type=float, default=0.7)
    parser.add_argument("--IPS_his_seq_len", type=int, default=20)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--trace_epoch", type=str2bool, default=True)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = akt_config(params)

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
    model = AKT(global_params, global_objects).to(global_params["device"])
    global_objects["models"]["kt_model"] = model
    trainer = KnowledgeTracingTrainer(global_params, global_objects)
    trainer.train()
