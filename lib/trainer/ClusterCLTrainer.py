import torch
import torch.nn as nn
import numpy as np

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .Cluster import Cluster
from .LossRecord import LossRecord


class ClusterCLTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(ClusterCLTrainer, self).__init__(params, objects)
        self.dataset_adv_generated = None
        self.num_epoch_adv_gen = 0
        self.adv_loss = LossRecord(["adv pred loss", "adv entropy", "adv mse loss"])
        self.clus = Cluster(params)

    def train(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        test_loader = self.objects["data_loaders"]["test_loader"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]
        cl_type = self.params["other"]["cluster_cl"]["cl_type"]
        max_entropy_aug_config = self.params["other"]["max_entropy_aug"]
        random_select_aug_len = self.params["other"]["cluster_cl"]["random_select_aug_len"]

        train_statics = train_loader.dataset.get_statics_kt_dataset()
        print(f"train, seq: {train_statics[0]}, sample: {train_statics[1]}, accuracy: {train_statics[2]:<.4}")
        if train_strategy["type"] == "valid_test":
            valid_statics = self.objects["data_loaders"]["valid_loader"].dataset.get_statics_kt_dataset()
            print(f"valid, seq: {valid_statics[0]}, sample: {valid_statics[1]}, accuracy: {valid_statics[2]:<.4}")
        test_statics = test_loader.dataset.get_statics_kt_dataset()
        print(f"test, seq: {test_statics[0]}, sample: {test_statics[1]}, accuracy: {test_statics[2]:<.4}")

        use_warm_up4cl = self.params["other"]["cluster_cl"]["use_warm_up4cl"]
        epoch_warm_up4cl = self.params["other"]["cluster_cl"]["epoch_warm_up4cl"]
        epoch_warm_up4online_sim = self.params["other"]["cluster_cl"]["epoch_warm_up4online_sim"]

        for epoch in range(1, num_epoch + 1):
            self.do_online_sim()
            self.do_max_entropy_aug()
            self.do_cluster()

            # 有对抗样本后，随机增强只需要生成一个view
            use_adv_aug = max_entropy_aug_config["use_adv_aug"] and (epoch > epoch_warm_up4online_sim)
            do_cl = (not use_warm_up4cl) or (use_warm_up4cl and (epoch > epoch_warm_up4cl))
            if do_cl and use_adv_aug:
                dataset_config_this = self.params["datasets_config"]["train"]
                train_loader.dataset.set_use_aug()
                dataset_config_this["kt4aug"]["num_aug"] = 1
            if do_cl and not use_adv_aug:
                dataset_config_this = self.params["datasets_config"]["train"]
                train_loader.dataset.set_use_aug()
                dataset_config_this["kt4aug"]["num_aug"] = 2
            if not do_cl:
                train_loader.dataset.set_not_use_aug()

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                num_seq = batch["mask_seq"].shape[0]
                loss = 0.

                if do_cl:
                    weight_cl_loss = self.params["loss_config"]["cl loss"]
                    if use_adv_aug:
                        cl_loss = model.get_cluster_cl_loss_adv(batch, self.clus, self.dataset_adv_generated)
                    else:
                        cl_loss = model.get_cluster_cl_loss_one_seq(batch, self.clus, cl_type, random_select_aug_len)
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
            latent_all = []
            model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    if random_select_aug_len:
                        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
                        latent = model.get_latent(batch)
                        latent = latent[:, 2:][mask_bool_seq[:, 2:]]
                    elif cl_type == "last_time":
                        latent = model.get_latent_last(batch)
                    elif cl_type == "mean_pool":
                        latent = model.get_latent_mean(batch)
                    else:
                        raise NotImplementedError()
                    latent_all.append(latent)
            latent_all = np.array(torch.cat(latent_all, dim=0).detach().cpu().tolist())
            self.clus.train(latent_all)

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
            concept_emb = model.get_concept_emb()
            train_loader.dataset.online_similarity.analysis(concept_emb)

    def do_max_entropy_aug(self):
        max_entropy_aug_config = self.params["other"]["max_entropy_aug"]
        use_adv_aug = max_entropy_aug_config["use_adv_aug"]
        epoch_warm_up4online_sim = self.params["other"]["cluster_cl"]["epoch_warm_up4online_sim"]
        current_epoch = self.train_record.get_current_epoch()
        epoch_interval_generate = max_entropy_aug_config["epoch_interval_generate"]
        loop_adv = max_entropy_aug_config["loop_adv"]
        epoch_generate = max_entropy_aug_config["epoch_generate"]
        adv_learning_rate = max_entropy_aug_config["adv_learning_rate"]
        eta = max_entropy_aug_config["eta"]
        gamma = max_entropy_aug_config["gamma"]

        do_generate = ((current_epoch - epoch_warm_up4online_sim) % epoch_interval_generate == 0)
        do_generate = use_adv_aug and do_generate and (self.num_epoch_adv_gen < epoch_generate)
        model = self.objects["models"]["kt_model"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        if do_generate and (current_epoch >= epoch_warm_up4online_sim):
            model.eval()
            train_loader.dataset.set_not_use_aug()
            # RNN就需要加上torch.backends.cudnn.enabled = False，才能在eval模式下通过网络还能保留梯度
            torch.backends.cudnn.enabled = False

            data_generated = {
                "seq_id": [],
                "emb_seq": []
            }
            for batch in train_loader:
                num_seq = batch["mask_seq"].shape[0]
                inputs_max, adv_predict_loss, adv_entropy, adv_mse_loss = (
                    model.get_max_entropy_adv_aug_emb(batch, adv_learning_rate, loop_adv, eta, gamma))
                self.adv_loss.add_loss("adv pred loss", adv_predict_loss * num_seq, num_seq)
                self.adv_loss.add_loss("adv entropy", adv_entropy * num_seq, num_seq)
                self.adv_loss.add_loss("adv mse loss", adv_mse_loss * num_seq, num_seq)
                data_generated["seq_id"].append(batch["seq_id"].to("cpu"))
                data_generated["emb_seq"].append(inputs_max.detach().clone().to("cpu"))

            print(self.adv_loss.get_str())
            self.adv_loss.clear_loss()
            for k in data_generated:
                data_generated[k] = torch.cat(data_generated[k], dim=0)
            self.save_adv_data(data_generated)

            train_loader.dataset.set_use_aug()
            torch.backends.cudnn.enabled = True

    def save_adv_data(self, data_adv):
        train_dataset = self.objects["data_loaders"]["train_loader"].dataset
        seq_len, dim_emb = data_adv["emb_seq"].shape[1], data_adv["emb_seq"].shape[2]
        if self.dataset_adv_generated is None:
            self.dataset_adv_generated = {
                "emb_seq": torch.empty((len(train_dataset), seq_len, dim_emb), dtype=torch.float, device="cpu")
            }

        for k in data_adv.keys():
            if k != "seq_id":
                self.dataset_adv_generated[k][data_adv["seq_id"]] = data_adv[k]
