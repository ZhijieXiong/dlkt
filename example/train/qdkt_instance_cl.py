import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from qdkt_config import qdkt_instance_cl_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.dataset.KTDataset4Aug import KTDataset4Aug
from lib.sequential_model.qDKT import qDKT
from lib.trainer.InstanceCLTrainer import InstanceCLTrainer


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
    parser.add_argument("--num_epoch", type=int, default=200)
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
    # instance cl参数
    parser.add_argument("--temp", type=float, default=0.01)
    parser.add_argument("--weight_cl_loss", type=float, default=0.01)
    parser.add_argument("--use_weight_dynamic", type=str2bool, default=True)
    parser.add_argument("--weight_dynamic_type", type=str, default="linear_increase",
                        choices=("multi_step", "linear_increase"))
    parser.add_argument("--multi_step_weight", type=str,
                        default="[(5, 0.001), (10, 0.005), (20, 0.01), (30, 0.05), (40, 0.1)]")
    parser.add_argument("--linear_increase_epoch", type=int, default=1)
    parser.add_argument("--linear_increase_value", type=float, default=0.003)
    parser.add_argument("--latent_type4cl", type=str, default="last_time",
                        choices=("last_time", "all_time", "mean_pool"))
    parser.add_argument("--use_emb_dropout4cl", type=str2bool, default=True)
    parser.add_argument("--emb_dropout4cl", type=float, default=0.3)
    parser.add_argument("--data_aug_type4cl", type=str, default="hybrid",
                        choices=("original_data_aug", "model_aug", "hybrid"))
    parser.add_argument("--use_online_sim", type=str2bool, default=True)
    parser.add_argument("--use_warm_up4online_sim", type=str2bool, default=True)
    parser.add_argument("--epoch_warm_up4online_sim", type=float, default=4)
    # data aug参数
    parser.add_argument("--aug_type", type=str, default="informative_aug",
                        choices=("random_aug", "informative_aug"))
    parser.add_argument("--use_random_select_aug_len", type=str2bool, default=True)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--insert_prob", type=float, default=0.2)
    parser.add_argument("--replace_prob", type=float, default=0.3)
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
    parser.add_argument("--loop_adv", type=int, default=5)
    parser.add_argument("--epoch_generate", type=int, default=40)
    parser.add_argument("--adv_learning_rate", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = qdkt_instance_cl_config(params)

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
    trainer = InstanceCLTrainer(global_params, global_objects)
    trainer.train()

    # LMO as09 domain 6, use online sim, use warm up 4 online sim (4), random aug len, use hard neg, crop: 0.1, replace: 0.3, insert: 0.3, last time, weight cl: 0.1
    # best valid epoch: 26  , best test epoch: 10
    # train performance by best valid epoch is main metric: 0.9144   , AUC: 0.9144   , ACC: 0.83542  , RMSE: 0.3358   , MAE: 0.24055  ,
    # valid performance by best valid epoch is main metric: 0.83598  , AUC: 0.83598  , ACC: 0.80159  , RMSE: 0.36995  , MAE: 0.2612   ,
    # test performance by best valid epoch is main metric: 0.74825  , AUC: 0.74825  , ACC: 0.69697  , RMSE: 0.44866  , MAE: 0.35422  ,
    # ----------------------------------------------------------------------------------------------------
    # train performance by best train epoch is main metric: 0.93723  , AUC: 0.93723  , ACC: 0.8602   , RMSE: 0.31316  , MAE: 0.21566  ,
    # test performance by best test epoch is main metric: 0.75813  , AUC: 0.75813  , ACC: 0.70747  , RMSE: 0.43868  , MAE: 0.36597  ,

    # LMO as09 domain 6, use online sim, use warm up 4 online sim (4), random aug len, use hard neg, crop: 0.1, replace: 0.3, insert: 0.3, last time, weight cl: 0.01
    # best valid epoch: 16  , best test epoch: 10
    # train performance by best valid epoch is main metric: 0.91123  , AUC: 0.91123  , ACC: 0.83412  , RMSE: 0.33866  , MAE: 0.24477  ,
    # valid performance by best valid epoch is main metric: 0.83555  , AUC: 0.83555  , ACC: 0.7985   , RMSE: 0.3723   , MAE: 0.26877  ,
    # test performance by best valid epoch is main metric: 0.75274  , AUC: 0.75274  , ACC: 0.6996   , RMSE: 0.44742  , MAE: 0.35539  ,
    # ----------------------------------------------------------------------------------------------------
    # train performance by best train epoch is main metric: 0.94697  , AUC: 0.94697  , ACC: 0.87282  , RMSE: 0.29994  , MAE: 0.19976  ,
    # test performance by best test epoch is main metric: 0.76326  , AUC: 0.76326  , ACC: 0.70373  , RMSE: 0.43997  , MAE: 0.36447  ,

    # LMO as09 domain 6, use online sim, use warm up 4 online sim (4), random aug len, use hard neg, crop: 0.1, replace: 0.3, insert: 0.3, mean pool, weight cl: 0.1
    # best valid epoch: 19  , best test epoch: 8
    # train performance by best valid epoch is main metric: 0.91952  , AUC: 0.91952  , ACC: 0.84227  , RMSE: 0.32967  , MAE: 0.22858  ,
    # valid performance by best valid epoch is main metric: 0.83044  , AUC: 0.83044  , ACC: 0.79938  , RMSE: 0.37376  , MAE: 0.2586   ,
    # test performance by best valid epoch is main metric: 0.75645  , AUC: 0.75645  , ACC: 0.70673  , RMSE: 0.4479   , MAE: 0.34415  ,
    # ----------------------------------------------------------------------------------------------------
    # train performance by best train epoch is main metric: 0.95398  , AUC: 0.95398  , ACC: 0.88157  , RMSE: 0.28926  , MAE: 0.18382  ,
    # test performance by best test epoch is main metric: 0.76453  , AUC: 0.76453  , ACC: 0.71154  , RMSE: 0.43696  , MAE: 0.3582   ,

    # LMO as09 domain 6, use online sim, use warm up 4 online sim (4), random aug len, use hard neg, crop: 0.1, replace: 0.3, insert: 0.3, mean pool, weight cl: 0.01
    # best valid epoch: 12  , best test epoch: 6
    # train performance by best valid epoch is main metric: 0.90836  , AUC: 0.90836  , ACC: 0.83372  , RMSE: 0.33876  , MAE: 0.23734  ,
    # valid performance by best valid epoch is main metric: 0.82965  , AUC: 0.82965  , ACC: 0.80105  , RMSE: 0.37276  , MAE: 0.26164  ,
    # test performance by best valid epoch is main metric: 0.75965  , AUC: 0.75965  , ACC: 0.70835  , RMSE: 0.44417  , MAE: 0.34575  ,
    # ----------------------------------------------------------------------------------------------------
    # train performance by best train epoch is main metric: 0.95443  , AUC: 0.95443  , ACC: 0.8839   , RMSE: 0.28731  , MAE: 0.18206  ,
    # test performance by best test epoch is main metric: 0.7649   , AUC: 0.7649   , ACC: 0.7143   , RMSE: 0.43566  , MAE: 0.35651  ,
