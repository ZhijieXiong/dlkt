import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from dimkt_config import dimkt_instance_cl_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.dataset.KTDataset4Aug import KTDataset4Aug
from lib.model.DIMKT import DIMKT
from lib.trainer.InstanceCLTrainer import InstanceCLTrainer


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
    parser.add_argument("--lr_schedule_step", type=int, default=10)
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
    # instance cl参数
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--weight_cl_loss", type=float, default=0.1)
    # warm up
    parser.add_argument("--use_warm_up4cl", type=str2bool, default=False)
    parser.add_argument("--epoch_warm_up4cl", type=float, default=2)
    # cl loss weight动态变化
    parser.add_argument("--use_weight_dynamic", type=str2bool, default=False)
    parser.add_argument("--weight_dynamic_type", type=str, default="multi_step",
                        choices=("multi_step", "linear_increase"))
    parser.add_argument("--multi_step_weight", type=str,
                        default="[[1, 0.1], [3, 0.03], [5, 0.01], [10, 0.0001], [200, 0.000001]]")
    parser.add_argument("--linear_increase_epoch", type=int, default=1)
    parser.add_argument("--linear_increase_value", type=float, default=0.1)
    parser.add_argument("--use_stop_cl_after", type=str2bool, default=False)
    parser.add_argument("--epoch_stop_cl", type=int, default=3)
    parser.add_argument("--latent_type4cl", type=str, default="last_time",
                        choices=("last_time", "all_time", "mean_pool"))
    # model aug参数
    parser.add_argument("--use_emb_dropout4cl", type=str2bool, default=True)
    parser.add_argument("--emb_dropout4cl", type=float, default=0.1)
    parser.add_argument("--data_aug_type4cl", type=str, default="original_data_aug",
                        choices=("original_data_aug", "model_aug", "hybrid"))
    # neg sample参数
    parser.add_argument("--use_neg", type=str2bool, default=True)
    parser.add_argument("--use_neg_filter", type=str2bool, default=True)
    parser.add_argument("--neg_sim_threshold", type=float, default=0.7, help="cos sim, between (0, 1)")
    # info aug参数
    parser.add_argument("--use_online_sim", type=str2bool, default=True)
    parser.add_argument("--use_warm_up4online_sim", type=str2bool, default=True)
    parser.add_argument("--epoch_warm_up4online_sim", type=float, default=4)
    # data aug参数
    parser.add_argument("--aug_type", type=str, default="informative_aug",
                        choices=("random_aug", "informative_aug"))
    parser.add_argument("--use_random_select_aug_len", type=str2bool, default=False)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--insert_prob", type=float, default=0.1)
    parser.add_argument("--replace_prob", type=float, default=0.1)
    parser.add_argument("--crop_prob", type=float, default=0.1)
    parser.add_argument("--permute_prob", type=float, default=0.1)
    parser.add_argument("--use_hard_neg", type=str2bool, default=False)
    parser.add_argument("--hard_neg_prob", type=float, default=1)
    parser.add_argument("--aug_order", type=str, default="['crop', 'replace', 'insert']",
                        help="CL4KT: ['mask', 'replace', 'permute', 'crop']"
                             "info aug: ['mask', 'crop', 'replace', 'insert']")
    parser.add_argument("--offline_sim_type", type=str, default="order",
                        choices=("order",))
    # max entropy adv aug参数
    parser.add_argument("--use_adv_aug", type=str2bool, default=False)
    parser.add_argument("--epoch_interval_generate", type=int, default=1)
    parser.add_argument("--loop_adv", type=int, default=3)
    parser.add_argument("--epoch_generate", type=int, default=40)
    parser.add_argument("--adv_learning_rate", type=float, default=20.0)
    parser.add_argument("--eta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = dimkt_instance_cl_config(params)

    if params["train_strategy"] == "valid_test":
        valid_params = deepcopy(global_params)
        valid_params["datasets_config"]["dataset_this"] = "valid"
        dataset_valid = KTDataset(valid_params, global_objects)
        dataloader_valid = DataLoader(dataset_valid, batch_size=params["evaluate_batch_size"], shuffle=False)
    else:
        dataloader_valid = None

    train_params = deepcopy(global_params)
    train_params["datasets_config"]["dataset_this"] = "train"
    dataset_train = KTDataset4Aug(train_params, global_objects)
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
    trainer = InstanceCLTrainer(global_params, global_objects)
    trainer.train()