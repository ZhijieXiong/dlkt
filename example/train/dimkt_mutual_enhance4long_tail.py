import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from dimkt_config import dimkt_mutual_enhance4long_tail_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.model.DIMKT import DIMKT
from lib.trainer.MutualEnhance4LongTailTrainer import MutualEnhance4LongTailTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="random_split_leave_multi_out_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2012")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, default="assist2012_train_split_5.txt")
    parser.add_argument("--valid_file_name", type=str, default="assist2012_valid_split_5.txt")
    parser.add_argument("--test_file_name", type=str, default="assist2012_test_split_5.txt")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam",
                        choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 训练策略
    parser.add_argument("--train_strategy", type=str, default="valid_test",
                        choices=("valid_test", "no_valid"))
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--epoch_early_stop", type=int, default=10)
    parser.add_argument("--use_last_average", type=str2bool, default=False)
    parser.add_argument("--epoch_last_average", type=int, default=5)
    parser.add_argument("--main_metric", type=str, default="AUC")
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--multi_metrics", type=str, default="[('AUC', 1), ('ACC', 1)]")
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=True)
    parser.add_argument("--lr_schedule_type", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step", type=int, default=5)
    parser.add_argument("--lr_schedule_milestones", type=str, default="[5]")
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
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
    # 是否使用LLM的emb初始化
    parser.add_argument("--use_LLM_emb4question", type=str2bool, default=False)
    parser.add_argument("--use_LLM_emb4concept", type=str2bool, default=False)
    parser.add_argument("--train_LLM_emb", type=str2bool, default=False)
    # 是否将head question的知识迁移到zero shot question
    parser.add_argument("--transfer_head2zero", type=str2bool, default=False)
    parser.add_argument("--head2tail_transfer_method", type=str, default="mean_pool",
                        choices=("mean_pool", ))
    # long tail设置
    parser.add_argument("--min_context_seq_len", type=int, default=10,
                        help="在构建question context时，只用长度大于min_context_seq_len的序列，以保证信息量的充足")
    parser.add_argument("--head_question_threshold", type=float, default=0.8)
    parser.add_argument("--head_seq_len", type=int, default=20,
                        help="只用序列长度大于head_seq_len的序列来训练seq branch，必须大于10")
    parser.add_argument("--dim_question", type=int, default=128)
    parser.add_argument("--dim_latent", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--use_transfer4seq", type=str2bool, default=True)
    parser.add_argument("--beta4transfer_seq", type=float, help="论文公式4中的beta", default=1)
    parser.add_argument("--gamma4transfer_question", type=float, help="论文公式8中的gamma", default=0)
    parser.add_argument("--only_update_low_fre", type=str2bool, default=True)
    parser.add_argument("--two_branch4question_transfer", type=str2bool, default=False)
    # 损失权重
    parser.add_argument("--weight_seq_loss", type=float, help="lambda U", default=0.1)
    parser.add_argument("--weight_question_loss", type=float, help="lambda I", default=0.1)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=True)
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

    model = DIMKT(global_params, global_objects).to(global_params["device"])
    global_objects["models"]["kt_model"] = model
    trainer = MutualEnhance4LongTailTrainer(global_params, global_objects)
    trainer.train()
