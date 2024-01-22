import torch
import torch.nn as nn
import numpy as np

from .BaseTrainer4ME_ADA import BaseTrainer4ME_ADA
from .Cluster import Cluster
from ..util.basic import *


class ClusterCLTrainer(BaseTrainer4ME_ADA):
    def __init__(self, params, objects):
        super(ClusterCLTrainer, self).__init__(params, objects)
        self.clus = Cluster(params)

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

        use_warm_up4cl = self.params["other"]["cluster_cl"]["use_warm_up4cl"]
        epoch_warm_up4cl = self.params["other"]["cluster_cl"]["epoch_warm_up4cl"]
        weight_cl_loss = self.params["loss_config"]["cl loss"]

        for epoch in range(1, num_epoch + 1):
            self.do_online_sim()
            self.do_max_entropy_aug()
            self.do_cluster()

            do_cluster_cl = (not use_warm_up4cl) or (use_warm_up4cl and epoch > epoch_warm_up4cl)
            if do_cluster_cl:
                train_loader.dataset.set_use_aug()
            else:
                train_loader.dataset.set_not_use_aug()

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                num_seq = batch["mask_seq"].shape[0]
                loss = 0.

                if do_cluster_cl:
                    cl_loss = model.get_cluster_cl_loss(batch, self.clus, self.params["other"]["cluster_cl"], self.dataset_adv_generated)
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

    def do_cluster(self):
        use_warm_up4cl = self.params["other"]["cluster_cl"]["use_warm_up4cl"]
        epoch_warm_up4cl = self.params["other"]["cluster_cl"]["epoch_warm_up4cl"]
        current_epoch = self.train_record.get_current_epoch()
        after_warm_up = current_epoch >= epoch_warm_up4cl
        model = self.objects["models"]["kt_model"]
        cl_type = self.params["other"]["cluster_cl"]["cl_type"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        random_select_aug_len = self.params["other"]["cluster_cl"]["random_select_aug_len"]

        if not use_warm_up4cl or after_warm_up:
            t_start = get_now_time()
            latent_all = []
            model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    if random_select_aug_len:
                        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
                        batch_size = mask_bool_seq.shape[0]
                        seq_len = mask_bool_seq.shape[1]
                        latent = model.get_latent(batch).detach()
                        if cl_type == "mean_pool":
                            with torch.no_grad():
                                mask4mean_pool = torch.ones_like(mask_bool_seq).to(self.params["device"])
                                mask4mean_pool = torch.cumsum(mask4mean_pool, dim=-1)
                                latent = torch.cumsum(latent, dim=-2) / mask4mean_pool.unsqueeze(2)
                        mask4select = torch.zeros(seq_len).to(self.params["device"])
                        mask4select[5::5] = 1
                        mask4select = mask4select.bool().repeat(batch_size, 1) & mask_bool_seq
                        latent = latent[:, 3:][mask4select[:, 3:]]
                    elif cl_type == "last_time":
                        latent = model.get_latent_last(batch).detach().cpu()
                    elif cl_type == "mean_pool":
                        latent = model.get_latent_mean(batch).detach().cpu()
                    else:
                        raise NotImplementedError()
                    latent_all.append(latent)
            # Cluster
            latent_all = np.array(torch.cat(latent_all, dim=0).tolist())

            # ClusterTorch
            # latent_all = torch.cat(latent_all, dim=0).detach().clone()
            # latent_all.requires_grad_(False)

            self.clus.train(latent_all)
            t_end = get_now_time()
            self.objects["logger"].info(f"cluster: from {t_start} to {t_end}")

    def do_online_sim(self):
        use_online_sim = self.params["other"]["cluster_cl"]["use_online_sim"]
        use_warm_up4online_sim = self.params["other"]["cluster_cl"]["use_warm_up4online_sim"]
        epoch_warm_up4online_sim = self.params["other"]["cluster_cl"]["epoch_warm_up4online_sim"]
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
