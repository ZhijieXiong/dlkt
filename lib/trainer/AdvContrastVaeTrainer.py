import torch.nn as nn

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer

from .LossRecord import LossRecord
from .util import *


class AdvContrastVaeTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def init_loss_record(self):
        used_losses = ["predict loss"]
        loss_config = self.params["loss_config"]
        for loss_name in loss_config:
            if loss_name in ["adv loss"]:
                used_losses.append(loss_name + " stage1")
                used_losses.append(loss_name + " stage2")
            else:
                used_losses.append(loss_name)
        return LossRecord(used_losses)

    def init_trainer(self):
        # 初始化optimizer和scheduler
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]

        kt_model = self.objects["models"]["kt_model"]
        contrastive_discriminator = self.objects["models"]["contrastive_discriminator"]
        adversary_discriminator = self.objects["models"]["adversary_discriminator"]

        kt_model_optimizer_config = self.params["optimizers_config"]["kt_model"]
        dual_optimizer_config = self.params["optimizers_config"]["dual"]
        prior_optimizer_config = self.params["optimizers_config"]["prior"]

        kt_model_scheduler_config = self.params["schedulers_config"]["kt_model"]
        dual_scheduler_config = self.params["schedulers_config"]["dual"]
        prior_scheduler_config = self.params["schedulers_config"]["prior"]

        optimizers["kt_model"] = create_optimizer(kt_model.parameters(), kt_model_optimizer_config)
        if kt_model_scheduler_config["use_scheduler"]:
            schedulers["kt_model"] = create_scheduler(optimizers["kt_model"], kt_model_scheduler_config)
        else:
            schedulers["kt_model"] = None

        dual_params = [{"params": kt_model.encoder_layer.parameters()}, {"params": contrastive_discriminator.parameters()}]
        optimizers["dual"] = create_optimizer(dual_params, dual_optimizer_config)
        if dual_scheduler_config["use_scheduler"]:
            schedulers["dual"] = create_scheduler(optimizers["dual"], dual_scheduler_config)
        else:
            schedulers["dual"] = None

        optimizers["prior"] = create_optimizer(adversary_discriminator.parameters(), prior_optimizer_config)
        if prior_scheduler_config["use_scheduler"]:
            schedulers["prior"] = create_scheduler(optimizers["prior"], prior_scheduler_config)
        else:
            schedulers["prior"] = None

    def train(self):
        train_strategy = self.params["train_strategy"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        kt_model = self.objects["models"]["kt_model"]
        kt_model_grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        kt_model_schedulers_config = self.params["schedulers_config"]["kt_model"]
        kt_model_optimizer = self.objects["optimizers"]["kt_model"]
        kt_model_scheduler = self.objects["schedulers"]["kt_model"]

        contrastive_discriminator = self.objects["models"]["contrastive_discriminator"]
        dual_net_grad_clip_config = self.params["grad_clip_config"]["dual"]
        dual_net_schedulers_config = self.params["schedulers_config"]["dual"]
        dual_net_optimizer = self.objects["optimizers"]["dual"]
        dual_net_scheduler = self.objects["schedulers"]["dual"]

        adversary_discriminator = self.objects["models"]["adversary_discriminator"]
        prior_net_grad_clip_config = self.params["grad_clip_config"]["prior"]
        prior_net_schedulers_config = self.params["schedulers_config"]["prior"]
        prior_net_optimizer = self.objects["optimizers"]["prior"]
        prior_net_scheduler = self.objects["schedulers"]["prior"]

        self.print_data_statics()

        cur_step = 1
        total_step = 20000
        for epoch in range(1, num_epoch + 1):
            kt_model.train()
            for batch_idx, batch in enumerate(train_loader):
                # --------------------------VAE---------------------------
                kt_model_optimizer.zero_grad()
                dual_net_optimizer.zero_grad()
                loss = kt_model.get_loss_stage1(batch, self.loss_record, cur_step/total_step)
                loss.backward()
                if kt_model_grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(kt_model.parameters(), max_norm=kt_model_grad_clip_config["grad_clipped"])
                kt_model_optimizer.step()
                if dual_net_grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(contrastive_discriminator.parameters(),
                                             max_norm=dual_net_grad_clip_config["grad_clipped"])
                dual_net_optimizer.step()

                # --------------------------ADV------------------------------
                prior_net_optimizer.zero_grad()
                adv_kl_loss = kt_model.get_loss_stage2(batch, self.loss_record)
                adv_kl_loss.backward()
                if prior_net_grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(adversary_discriminator.parameters(),
                                             max_norm=prior_net_grad_clip_config["grad_clipped"])
                prior_net_optimizer.step()

                cur_step += 1

            if kt_model_schedulers_config["use_scheduler"]:
                kt_model_scheduler.step()
            if dual_net_schedulers_config["use_scheduler"]:
                dual_net_scheduler.step()
            if prior_net_schedulers_config["use_scheduler"]:
                prior_net_scheduler.step()

            self.evaluate()

            if self.stop_train():
                break
