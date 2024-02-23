import argparse
import os
import torch
from copy import deepcopy
from torch.utils.data import DataLoader

from config.dimkt_config import dimkt_mutual_enhance4long_tail_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.model.DIMKT import DIMKT
from lib.trainer.MutualEnhance4LongTailTrainer import MutualEnhance4LongTailTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="our_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2012")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, default="assist2012_train_fold_0.txt")
    parser.add_argument("--valid_file_name", type=str, default="assist2012_valid_fold_0.txt")
    parser.add_argument("--test_file_name", type=str, default="assist2012_test_fold_0.txt")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0.0001)
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
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=False)
    parser.add_argument("--lr_schedule_type", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step", type=int, default=10)
    parser.add_argument("--lr_schedule_milestones", type=str, default="[5]")
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # DIMKT数据处理参数
    parser.add_argument("--num_min_question", type=int, default=25)
    parser.add_argument("--num_min_concept", type=int, default=30)
    # 模型参数
    parser.add_argument("--num_concept", type=int, default=265)
    parser.add_argument("--num_question", type=int, default=53091)
    parser.add_argument("--dim_emb", type=int, default=128)
    parser.add_argument("--num_question_diff", type=int, default=100)
    parser.add_argument("--num_concept_diff", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    # MELT设置
    parser.add_argument("--two_stage", type=str2bool, default=True)
    parser.add_argument("--dim_question", type=int, default=128)
    parser.add_argument("--dim_latent", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=200)
    # one stage: User branch和Item branch随KT model一起训练
    parser.add_argument("--min_context_seq_len", type=int, default=10,
                        help="在构建question context来训练item branch时，只用长度大于min_context_seq_len的序列，以保证信息量的充足")
    parser.add_argument("--head_question_threshold", type=float, default=0.8)
    parser.add_argument("--head_seq_len", type=int, default=20,
                        help="只用序列长度大于head_seq_len的序列来训练seq branch，必须大于10")
    parser.add_argument("--use_transfer4seq", type=str2bool, default=False, help="是否使用User branch")
    parser.add_argument("--beta4transfer_seq", type=float, help="论文公式4中的beta", default=1)
    parser.add_argument("--gamma4transfer_question", type=float, help="论文公式8中的gamma", default=0)
    parser.add_argument("--only_update_low_fre", type=str2bool, default=True)
    parser.add_argument("--two_branch4question_transfer", type=str2bool, default=True,
                        help="训练Item branch时是否区分做对和做错的context")
    # two stage: 先训练好KT model，再训练Item branch
    parser.add_argument("--save_model_dir", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\saved_models\observe\dimkt\2024-01-16@11-09-27@@DIMKT@@seed_0@@our_setting@@assist2012_train_fold_0")
    parser.add_argument("--optimizer_type_question_branch", type=str, default="adam",
                        choices=("adam", "sgd"))
    parser.add_argument("--weight_decay_question_branch", type=float, default=0)
    parser.add_argument("--momentum_question_branch", type=float, default=0.9)
    parser.add_argument("--learning_rate_question_branch", type=float, default=0.0001)
    parser.add_argument("--enable_lr_schedule_question_branch", type=str2bool, default=False)
    parser.add_argument("--lr_schedule_type_question_branch", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step_question_branch", type=int, default=10)
    parser.add_argument("--lr_schedule_milestones_question_branch", type=str, default="[5]")
    parser.add_argument("--lr_schedule_gamma_question_branch", type=float, default=0.5)
    parser.add_argument("--enable_clip_grad_question_branch", type=str2bool, default=True)
    parser.add_argument("--grad_clipped_question_branch", type=float, default=5.0)
    # 损失权重（只对one stage有效）
    parser.add_argument("--weight_seq_loss", type=float, help="lambda U", default=0.1)
    parser.add_argument("--weight_question_loss", type=float, help="lambda I", default=0.1)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = dimkt_mutual_enhance4long_tail_config(params)
    # global_params["device"] = "cpu"

    if params["train_strategy"] == "valid_test":
        valid_params = deepcopy(global_params)
        valid_params["datasets_config"]["dataset_this"] = "valid"
        dataset_valid = KTDataset(valid_params, global_objects)
        dataloader_valid = DataLoader(dataset_valid, batch_size=params["evaluate_batch_size"], shuffle=False)
    else:
        dataloader_valid = None

    train_params = deepcopy(global_params)
    train_params["datasets_config"]["dataset_this"] = "train"
    dataset_train = KTDataset(train_params, global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)

    test_params = deepcopy(global_params)
    test_params["datasets_config"]["dataset_this"] = "test"
    dataset_test = KTDataset(test_params, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)

    global_objects["data_loaders"]["train_loader"] = dataloader_train
    global_objects["data_loaders"]["valid_loader"] = dataloader_valid
    global_objects["data_loaders"]["test_loader"] = dataloader_test

    save_model_path = os.path.join(params["save_model_dir"], "kt_model.pth")
    if params["two_stage"] and os.path.exists(save_model_path):
        global_params["other"]["mutual_enhance4long_tail"]["train_kt"] = False
        global_params["other"]["mutual_enhance4long_tail"]["kt_model_path"] = save_model_path
        model = torch.load(save_model_path).to(global_params["device"])
        model.params = global_params
        model.objects = global_objects
    else:
        global_params["other"]["mutual_enhance4long_tail"]["train_kt"] = True
        model = DIMKT(global_params, global_objects).to(global_params["device"])
    global_objects["models"]["kt_model"] = model
    trainer = MutualEnhance4LongTailTrainer(global_params, global_objects)
    if not params["two_stage"]:
        trainer.train_one_stage()
    else:
        trainer.train_two_stage()
