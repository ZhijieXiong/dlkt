import torch
import torch.nn as nn

from .BaseTrainer4ME_ADA import BaseTrainer4ME_ADA
from ..util.basic import *


class InstanceCLTrainer(BaseTrainer4ME_ADA):
    def __init__(self, params, objects):
        super(InstanceCLTrainer, self).__init__(params, objects)

    def train(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        self.print_data_statics()

        weight_cl_loss = self.params["loss_config"]["cl loss"]
        instance_cl_params = self.params["other"]["instance_cl"]
        use_adv_aug = instance_cl_params["use_adv_aug"]
        latent_type4cl = instance_cl_params["latent_type4cl"]

        for epoch in range(1, num_epoch + 1):
            self.do_online_sim()
            if use_adv_aug:
                self.do_max_entropy_aug()

            train_loader.dataset.set_use_aug()

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                num_seq = batch["mask_seq"].shape[0]
                loss = 0.

                if latent_type4cl in ["mean_pool", "last_time"]:
                    cl_loss = model.get_instance_cl_loss(batch, instance_cl_params, self.dataset_adv_generated)
                elif latent_type4cl == "all_time":
                    cl_loss = model.get_instance_cl_loss_all_interaction(batch, instance_cl_params)
                else:
                    raise NotImplementedError()
                self.loss_record.add_loss("cl loss", cl_loss.detach().cpu().item() * num_seq, num_seq)
                loss = loss + weight_cl_loss * cl_loss

                predict_loss = model.get_predict_loss(batch)
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
                loss = loss + predict_loss

                loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])

                optimizer.step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break

    def do_online_sim(self):
        use_online_sim = self.params["other"]["instance_cl"]["use_online_sim"]
        use_warm_up4online_sim = self.params["other"]["instance_cl"]["use_warm_up4online_sim"]
        epoch_warm_up4online_sim = self.params["other"]["instance_cl"]["epoch_warm_up4online_sim"]
        current_epoch = self.train_record.get_current_epoch()
        after_warm_up = current_epoch >= epoch_warm_up4online_sim
        dataset_config_this = self.params["datasets_config"]["train"]
        aug_type = dataset_config_this["kt4aug"]["aug_type"]

        model = self.objects["models"]["kt_model"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        if aug_type == "informative_aug" and use_online_sim and (not use_warm_up4online_sim or after_warm_up):
            t_start = get_now_time()
            concept_emb = model.get_concept_emb_all()
            train_loader.dataset.online_similarity.analysis(concept_emb)
            t_end = get_now_time()
            self.objects["logger"].info(f"online similarity analysis: from {t_start} to {t_end}")
