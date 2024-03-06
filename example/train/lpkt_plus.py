import argparse
from copy import deepcopy

from torch.utils.data import DataLoader

from config.lpkt_config import lpkt_plus_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset4LPKTPlus import KTDataset4LPKTPlus
from lib.dataset.KTDataset import KTDataset
from lib.model.LPKTPlus_dev import LPKTPlus
from lib.trainer.CognitionTracingTrainer import CognitionTracingTrainer


# def collect_fn(batch):
#     elem = batch[0]
#     batch_size = len(batch)
#     batch_data = {}
#     for k in elem.keys():
#         batch_data[k] = torch.concat([batch[i][k].unsqueeze(0) for i in range(batch_size)], dim=0)
#     question_unique = torch.unique(batch_data["question_seq"])
#     return batch_data, question_unique


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
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 训练策略
    parser.add_argument("--train_strategy", type=str, default="valid_test", choices=("valid_test", "no_valid"))
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
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=False)
    parser.add_argument("--lr_schedule_type", type=str, default="StepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step", type=int, default=20)
    parser.add_argument("--lr_schedule_milestones", type=str, default="[5]")
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=1024)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 模型参数
    parser.add_argument("--num_concept", type=int, default=27)
    parser.add_argument("--num_question", type=int, default=1223)
    parser.add_argument("--dim_question", type=int, default=64)
    parser.add_argument("--dim_latent", type=int, default=64)
    parser.add_argument("--dim_correct", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    # 生成伪标签的参数
    parser.add_argument("--min_fre4diff", type=int, default=50)
    parser.add_argument("--min_fre4disc", type=int, default=50)
    parser.add_argument("--min_seq_len4disc", type=int, default=20)
    parser.add_argument("--percent_threshold", type=float, default=0.32, help="计算区分度时，选择正确率最高的k%和最低的k%序列")
    # 消融
    parser.add_argument("--ablation_set", type=int, default=1,
                        help="0: use time seq and interval time seq"
                             "1: only interval time seq"
                             "2: do not use time information")
    parser.add_argument("--use_init_weight", type=str2bool, default=False, help="是否使用基于IRT的参数初始化")
    # 损失权重
    parser.add_argument("--w_que_diff_pred", type=float, default=0)
    parser.add_argument("--w_que_disc_pred", type=float, default=0)
    parser.add_argument("--w_penalty_neg", type=float, default=0,
                        help="计算最终得分时，对于做对的题，惩罚ability-difficulty小于0（对应知识点）")
    parser.add_argument("--w_user_ability_pred", type=float, default=0)
    parser.add_argument("--w_learning", type=float, default=0)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = lpkt_plus_config(params)

    if params["train_strategy"] == "valid_test":
        valid_params = deepcopy(global_params)
        valid_params["datasets_config"]["dataset_this"] = "valid"
        dataset_valid = KTDataset(valid_params, global_objects)
        dataloader_valid = DataLoader(dataset_valid, batch_size=params["evaluate_batch_size"], shuffle=False)
    else:
        dataloader_valid = None

    train_params = deepcopy(global_params)
    train_params["datasets_config"]["dataset_this"] = "train"
    dataset_train = KTDataset4LPKTPlus(train_params, global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)

    test_params = deepcopy(global_params)
    test_params["datasets_config"]["dataset_this"] = "test"
    dataset_test = KTDataset(test_params, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {}
    global_objects["data_loaders"]["train_loader"] = dataloader_train
    global_objects["data_loaders"]["valid_loader"] = dataloader_valid
    global_objects["data_loaders"]["test_loader"] = dataloader_test

    model = LPKTPlus(global_params, global_objects).to(global_params["device"])
    global_objects["models"] = {}
    global_objects["models"]["kt_model"] = model
    trainer = CognitionTracingTrainer(global_params, global_objects)
    trainer.train()
