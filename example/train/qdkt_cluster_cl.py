import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from qdkt_config import qdkt_cluster_cl_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.dataset.KTDataset4Aug import KTDataset4Aug
from lib.model.qDKT import qDKT
from lib.trainer.ClusterCLTrainer import ClusterCLTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="random_split_leave_multi_out_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--data_type", type=str, default="multi_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_split_6.txt")
    parser.add_argument("--valid_file_name", type=str, default="assist2009_valid_split_6.txt")
    parser.add_argument("--test_file_name", type=str, default="assist2009_test_split_6.txt")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam",
                        choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 训练策略
    parser.add_argument("--train_strategy", type=str, default="valid_test",
                        choices=("valid_test", "no_valid"))
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--epoch_early_stop", type=int, default=10)
    parser.add_argument("--use_last_average", type=str2bool, default=False)
    parser.add_argument("--epoch_last_average", type=int, default=5)
    parser.add_argument("--main_metric", type=str, default="AUC")
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--multi_metrics", type=str, default="[('AUC', 1), ('ACC', 1)]")
    parser.add_argument("--learning_rate", type=float, default=0.001)
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
    # 模型参数
    parser.add_argument("--num_concept", type=int, default=123)
    parser.add_argument("--num_question", type=int, default=17751)
    parser.add_argument("--dim_concept", type=int, default=64)
    parser.add_argument("--dim_question", type=int, default=64)
    parser.add_argument("--dim_correct", type=int, default=128)
    parser.add_argument("--dim_latent", type=int, default=128)
    parser.add_argument("--rnn_type", type=str, default="gru")
    parser.add_argument("--num_rnn_layer", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_predict_layer", type=int, default=3)
    parser.add_argument("--dim_predict_mid", type=int, default=128)
    parser.add_argument("--activate_type", type=str, default="relu")
    # cluster CL参数（对比学习）
    parser.add_argument("--num_cluster", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--weight_cl_loss", type=float, default=0.3)
    parser.add_argument("--use_warm_up4cl", type=str2bool, default=True)
    parser.add_argument("--epoch_warm_up4cl", type=float, default=4)
    parser.add_argument("--use_online_sim", type=str2bool, default=True)
    parser.add_argument("--use_warm_up4online_sim", type=str2bool, default=True)
    parser.add_argument("--epoch_warm_up4online_sim", type=float, default=4)
    parser.add_argument("--cl_type", type=str, default="mean_pool",
                        choices=("last_time", "mean_pool"))
    # random aug和informative aug参数
    parser.add_argument("--aug_type", type=str, default="informative_aug",
                        choices=("random_aug", "informative_aug"))
    parser.add_argument("--use_random_select_aug_len", type=str2bool, default=True)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--insert_prob", type=float, default=0.2)
    parser.add_argument("--replace_prob", type=float, default=0.3)
    parser.add_argument("--crop_prob", type=float, default=0.1)
    parser.add_argument("--permute_prob", type=float, default=0.1)
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
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = qdkt_cluster_cl_config(params)

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

    model = qDKT(global_params, global_objects).to(global_params["device"])
    global_objects["models"]["kt_model"] = model
    trainer = ClusterCLTrainer(global_params, global_objects)
    trainer.train()

    # 0.0005, [5, 10]
    # temp: 0.01 weight: 0.1 cluster: 16
    # AUC: 0.83834  , ACC: 0.80322  , RMSE: 0.36876  , MAE: 0.26343
    # AUC: 0.75014  , ACC: 0.70175  , RMSE: 0.44868  , MAE: 0.35168
    # AUC: 0.75481  , ACC: 0.70649  , RMSE: 0.44183  , MAE: 0.35942

    # temp: 0.01 weight: 0.1 cluster: 32
    # AUC: 0.83716  , ACC: 0.80406  , RMSE: 0.3682   , MAE: 0.26537
    # AUC: 0.75247  , ACC: 0.70354  , RMSE: 0.44625  , MAE: 0.35234
    # AUC: 0.75768  , ACC: 0.7088   , RMSE: 0.44151  , MAE: 0.35408

    # # temp: 0.01 weight: 0.01 cluster: 32

    # temp: 0.01 weight: 0.1 cluster: 64
    # AUC: 0.83705  , ACC: 0.80075  , RMSE: 0.37074  , MAE: 0.26477
    # AUC: 0.75036  , ACC: 0.70211  , RMSE: 0.4496   , MAE: 0.35056
    # AUC: 0.75531  , ACC: 0.7086   , RMSE: 0.44134  , MAE: 0.35766

    # temp: 0.01 weight: 0.1 cluster: 128
    # AUC: 0.83491  , ACC: 0.80322  , RMSE: 0.36912  , MAE: 0.2691
    # AUC: 0.75366  , ACC: 0.70423  , RMSE: 0.44443  , MAE: 0.35405
    # AUC: 0.75768  , ACC: 0.70796  , RMSE: 0.44022  , MAE: 0.35639

    # temp: 0.01 weight: 0.1 cluster: 256
    # AUC: 0.83353  , ACC: 0.80346  , RMSE: 0.37001  , MAE: 0.26929
    # AUC: 0.75377  , ACC: 0.70608  , RMSE: 0.44445  , MAE: 0.35276
    # AUC: 0.75683  , ACC: 0.70833  , RMSE: 0.44085  , MAE: 0.3555

    # temp: 0.01 weight: 0.3 cluster: 512
    # AUC: 0.83098  , ACC: 0.80219  , RMSE: 0.37066  , MAE: 0.26467
    # AUC: 0.75355  , ACC: 0.7054   , RMSE: 0.44564  , MAE: 0.34991
    # AUC: 0.75493  , ACC: 0.70685  , RMSE: 0.44112  , MAE: 0.36104

    # 0.001, [5]
    # temp: 0.01 weight: 0.1 cluster: 32
    # AUC: 0.83857  , ACC: 0.80598  , RMSE: 0.36845  , MAE: 0.25611
    # AUC: 0.75213  , ACC: 0.70443  , RMSE: 0.44931  , MAE: 0.34655
    # AUC: 0.75754  , ACC: 0.70849  , RMSE: 0.44063  , MAE: 0.35714
