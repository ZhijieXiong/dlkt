import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from qdkt_config import qdkt_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.model.qDKT import qDKT
from lib.trainer.KnowledgeTracingTrainer import KnowledgeTracingTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, default="ednet-kt1_train_fold_0.txt")
    parser.add_argument("--valid_file_name", type=str, default="ednet-kt1_valid_fold_0.txt")
    parser.add_argument("--test_file_name", type=str, default="ednet-kt1_test.txt")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam",
                        choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 训练策略
    parser.add_argument("--train_strategy", type=str, default="valid_test",
                        choices=("valid_test", "no_valid"))
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--epoch_early_stop", type=int, default=10)
    parser.add_argument("--use_last_average", type=str2bool, default=True)
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
    parser.add_argument("--num_concept", type=int, default=1462)
    parser.add_argument("--num_question", type=int, default=11858)
    parser.add_argument("--dim_concept", type=int, default=64)
    parser.add_argument("--dim_question", type=int, default=64)
    parser.add_argument("--dim_correct", type=int, default=128)
    parser.add_argument("--dim_latent", type=int, default=128)
    parser.add_argument("--rnn_type", type=str, default="gru",
                        choices=("rnn", "lstm", "gru"))
    parser.add_argument("--num_rnn_layer", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_predict_layer", type=int, default=3)
    parser.add_argument("--dim_predict_mid", type=int, default=128)
    parser.add_argument("--activate_type", type=str, default="relu")
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = qdkt_config(params)

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

    model = qDKT(global_params, global_objects).to(global_params["device"])
    global_objects["models"]["kt_model"] = model
    trainer = KnowledgeTracingTrainer(global_params, global_objects)
    trainer.train()

    # assist2009
    # domain 0
    # AUC: 0.81119  , ACC: 0.79384  , RMSE: 0.37954  , MAE: 0.27214
    # AUC: 0.77465  , ACC: 0.74729  , RMSE: 0.41712  , MAE: 0.31898
    # AUC: 0.77684  , ACC: 0.75089  , RMSE: 0.41347  , MAE: 0.32403

    # domain 6
    # AUC: 0.82585  , ACC: 0.79635  , RMSE: 0.37493  , MAE: 0.27575
    # AUC: 0.75559  , ACC: 0.70579  , RMSE: 0.44308  , MAE: 0.35698
    # AUC: 0.76119  , ACC: 0.71178  , RMSE: 0.43823  , MAE: 0.35405

    # domain 8
    # AUC: 0.8208   , ACC: 0.79526  , RMSE: 0.37639  , MAE: 0.27545
    # AUC: 0.7795   , ACC: 0.75079  , RMSE: 0.41285  , MAE: 0.321
    # AUC: 0.78009  , ACC: 0.75401  , RMSE: 0.41033  , MAE: 0.32242

    # assist2012
    # domain 3
    # AUC: 0.7519   , ACC: 0.74583  , RMSE: 0.41656  , MAE: 0.33534
    # AUC: 0.73136  , ACC: 0.72575  , RMSE: 0.43021  , MAE: 0.34996
    # AUC: 0.73272  , ACC: 0.72827  , RMSE: 0.42791  , MAE: 0.35358

    # domain 5
    # AUC: 0.75289  , ACC: 0.74444  , RMSE: 0.41755  , MAE: 0.33546
    # AUC: 0.72747  , ACC: 0.73588  , RMSE: 0.42485  , MAE: 0.34082
    # AUC: 0.72928  , ACC: 0.73765  , RMSE: 0.42284  , MAE: 0.34624

    # domain 7
    # AUC: 0.75361  , ACC: 0.74894  , RMSE: 0.41435  , MAE: 0.33418
    # AUC: 0.72822  , ACC: 0.72162  , RMSE: 0.43225  , MAE: 0.35816
    # AUC: 0.72857  , ACC: 0.7218   , RMSE: 0.43171  , MAE: 0.35931
