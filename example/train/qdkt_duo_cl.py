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
    parser.add_argument("--use_early_stop", type=str2bool, default=False)
    parser.add_argument("--epoch_early_stop", type=int, default=10)
    parser.add_argument("--use_last_average", type=str2bool, default=True)
    parser.add_argument("--epoch_last_average", type=int, default=5)
    parser.add_argument("--main_metric", type=str, default="AUC")
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--multi_metrics", type=str, default="[('AUC', 1), ('ACC', 1)]")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=False)
    parser.add_argument("--lr_schedule_type", type=str, default="StepLR",
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
    parser.add_argument("--weight_cl_loss", type=float, default=1)
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

    # qdkt: valid performance by best valid epoch is main metric: 0.82321  , AUC: 0.82321  , ACC: 0.7986   , RMSE: 0.37518  , MAE: 0.25932  ,
    # test performance by best valid epoch is main metric: 0.75265  , AUC: 0.75265  , ACC: 0.70112  , RMSE: 0.44902  , MAE: 0.34723  ,

    # 0.1: valid performance by best valid epoch is main metric: 0.81771  , AUC: 0.81771  , ACC: 0.79763  , RMSE: 0.37802  , MAE: 0.26133  ,
    # test performance by best valid epoch is main metric: 0.75061  , AUC: 0.75061  , ACC: 0.70207  , RMSE: 0.45023  , MAE: 0.34648  ,

    # 0.01: valid performance by best valid epoch is main metric: 0.81569  , AUC: 0.81569  , ACC: 0.79841  , RMSE: 0.37707  , MAE: 0.26516  ,
    # test performance by best valid epoch is main metric: 0.75408  , AUC: 0.75408  , ACC: 0.70545  , RMSE: 0.44591  , MAE: 0.34876  ,

    # 0.001 (0.1): valid performance by best valid epoch is main metric: 0.81967  , AUC: 0.81967  , ACC: 0.79677  , RMSE: 0.37643  , MAE: 0.27156  ,
    # test performance by best valid epoch is main metric: 0.75315  , AUC: 0.75315  , ACC: 0.70342  , RMSE: 0.44486  , MAE: 0.35334  ,

    # 0.001 (0.05): valid performance by best valid epoch is main metric: 0.81961  , AUC: 0.81961  , ACC: 0.7966   , RMSE: 0.37643  , MAE: 0.27184  ,
    # test performance by best valid epoch is main metric: 0.75252  , AUC: 0.75252  , ACC: 0.70402  , RMSE: 0.44497  , MAE: 0.35388  ,

    # 0.1 (0.05): valid performance by best valid epoch is main metric: 0.82101  , AUC: 0.82101  , ACC: 0.79886  , RMSE: 0.37579  , MAE: 0.26533  ,
    # test performance by best valid epoch is main metric: 0.75621  , AUC: 0.75621  , ACC: 0.70488  , RMSE: 0.44518  , MAE: 0.34853  ,

    # 1 (0.05): valid performance by best valid epoch is main metric: 0.82443  , AUC: 0.82443  , ACC: 0.79814  , RMSE: 0.37546  , MAE: 0.26324  ,
    # test performance by best valid epoch is main metric: 0.75122  , AUC: 0.75122  , ACC: 0.70214  , RMSE: 0.44949  , MAE: 0.34801  ,
