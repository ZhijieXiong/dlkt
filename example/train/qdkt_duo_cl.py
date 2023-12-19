import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from qdkt_config import qdkt_duo_cl_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.dataset.KTDataset4Aug import KTDataset4Aug
from lib.model.qDKT import qDKT
from lib.trainer.DuoCLTrainer import DuoCLTrainer


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
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=True)
    parser.add_argument("--lr_schedule_type", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step", type=int, default=10)
    parser.add_argument("--lr_schedule_milestones", type=str, default="[5, 10]")
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 模型参数
    parser.add_argument("--num_concept", type=int, default=265)
    parser.add_argument("--num_question", type=int, default=53091)
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
    # duo参数（对比学习）
    parser.add_argument("--temp", type=float, default=0.01)
    parser.add_argument("--cl_type", type=str, default="mean_pool",
                        choices=("last_time", "mean_pool"))
    parser.add_argument("--weight_cl_loss", type=float, default=0.5)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = qdkt_duo_cl_config(params)

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
    trainer = DuoCLTrainer(global_params, global_objects)
    trainer.train()

    # temp: 0.01, weight: 0.01
    # AUC: 0.83285  , ACC: 0.80513  , RMSE: 0.37069  , MAE: 0.25545
    # AUC: 0.75358  , ACC: 0.70416  , RMSE: 0.44996  , MAE: 0.34471
    # AUC: 0.76353  , ACC: 0.71108  , RMSE: 0.43925  , MAE: 0.3522

    # temp: 0.01, weight: 0.1
    # AUC: 0.83509  , ACC: 0.80345  , RMSE: 0.37089  , MAE: 0.2553
    # AUC: 0.75521  , ACC: 0.70492  , RMSE: 0.4511   , MAE: 0.34229
    # AUC: 0.76264  , ACC: 0.71094  , RMSE: 0.44056  , MAE: 0.35121

    # temp: 0.01, weight: 0.3
    # AUC: 0.83147  , ACC: 0.80381  , RMSE: 0.37036  , MAE: 0.26171
    # AUC: 0.75624  , ACC: 0.70515  , RMSE: 0.44571  , MAE: 0.34899
    # AUC: 0.75885  , ACC: 0.70697  , RMSE: 0.44147  , MAE: 0.35562

    # temp: 0.01, weight: 0.5
    # AUC: 0.82977  , ACC: 0.79945  , RMSE: 0.37317  , MAE: 0.26569
    # AUC: 0.75293  , ACC: 0.70168  , RMSE: 0.44845  , MAE: 0.35129
    # AUC: 0.75417  , ACC: 0.70407  , RMSE: 0.4444   , MAE: 0.35624
