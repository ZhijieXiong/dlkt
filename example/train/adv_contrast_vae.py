import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from adv_contrast_vae_config import adv_contrast_vae_gru_config

from lib.util.parse import str2bool
from lib.util.set_up import set_seed
from lib.dataset.KTDataset import KTDataset
from lib.model.AC_VAE_GRU import AC_VAE_GRU
from lib.model.Module.AC_VAE import ContrastiveDiscriminator, AdversaryDiscriminator
from lib.trainer.AdvContrastVaeTrainer import AdvContrastVaeTrainer


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
    parser.add_argument("--enable_lr_schedule", type=str2bool, default=True)
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
    # 其它优化器的参数
    # dual: 优化的是kt model的encoder和ContrastiveDiscriminator
    parser.add_argument("--optimizer_type_dual", type=str, default="sgd",
                        choices=("adam", "sgd"))
    parser.add_argument("--weight_decay_dual", type=float, default=0.1)
    parser.add_argument("--momentum_dual", type=float, default=0.9)
    parser.add_argument("--learning_rate_dual", type=float, default=0.0003)
    parser.add_argument("--enable_lr_schedule_dual", type=str2bool, default=False)
    parser.add_argument("--lr_schedule_type_dual", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step_dual", type=int, default=10)
    parser.add_argument("--lr_schedule_milestones_dual", type=str, default="[5]")
    parser.add_argument("--lr_schedule_gamma_dual", type=float, default=0.5)
    parser.add_argument("--enable_clip_grad_dual", type=str2bool, default=False)
    parser.add_argument("--grad_clipped_dual", type=float, default=10.0)
    # prior: 优化的是AdversaryDiscriminator，即对抗VAE中用于生成latent的网络（原始VAE是用网络生成高斯分布的均值方差，然后重参数采样出来latent的）
    parser.add_argument("--optimizer_type_prior", type=str, default="sgd",
                        choices=("adam", "sgd"))
    parser.add_argument("--weight_decay_prior", type=float, default=0.1)
    parser.add_argument("--momentum_prior", type=float, default=0.9)
    parser.add_argument("--learning_rate_prior", type=float, default=0.0005)
    parser.add_argument("--enable_lr_schedule_prior", type=str2bool, default=False)
    parser.add_argument("--lr_schedule_type_prior", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--lr_schedule_step_prior", type=int, default=10)
    parser.add_argument("--lr_schedule_milestones_prior", type=str, default="[5]")
    parser.add_argument("--lr_schedule_gamma_prior", type=float, default=0.5)
    parser.add_argument("--enable_clip_grad_prior", type=str2bool, default=False)
    parser.add_argument("--grad_clipped_prior", type=float, default=10.0)
    # 损失权重参数
    parser.add_argument("--weight_kl_loss", type=float, default=0.005)
    parser.add_argument("--weight_cl_loss", type=float, default=0.01)
    parser.add_argument("--use_anneal", type=str2bool, default=True)
    # 消融
    parser.add_argument("--use_vae", type=str2bool, default=True)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = adv_contrast_vae_gru_config(params)

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

    kt_model = AC_VAE_GRU(global_params, global_objects).to(global_params["device"])
    contrastive_discriminator = ContrastiveDiscriminator(global_params, global_objects).to(global_params["device"])
    adversary_discriminator = AdversaryDiscriminator(global_params, global_objects).to(global_params["device"])
    global_objects["models"]["kt_model"] = kt_model
    global_objects["models"]["contrastive_discriminator"] = contrastive_discriminator
    global_objects["models"]["adversary_discriminator"] = adversary_discriminator
    trainer = AdvContrastVaeTrainer(global_params, global_objects)
    trainer.train()

