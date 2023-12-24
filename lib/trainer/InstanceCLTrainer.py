import torch
import torch.nn as nn
import torch.optim as optim

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .LossRecord import LossRecord
from ..model.Module.KTEmbedLayer import KTEmbedLayer
from ..util.basic import *


class InstanceCLTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(InstanceCLTrainer, self).__init__(params, objects)
        self.dataset_adv_generated = None
        self.num_epoch_adv_gen = 0
        self.adv_loss = LossRecord(["gen pred loss", "gen entropy loss", "gen mse loss"])

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
        instance_cl_config = self.params["other"]["instance_cl"]
        latent_type4cl = instance_cl_config["latent_type4cl"]
        use_warm_up4cluster_cl = self.params["other"]["instance_cl"]["use_warm_up4cluster_cl"]
        epoch_warm_up4cluster_cl = self.params["other"]["instance_cl"]["epoch_warm_up4cluster_cl"]
        use_stop_cl_after = self.params["other"]["instance_cl"]["use_stop_cl_after"]
        epoch_stop_cl = self.params["other"]["instance_cl"]["epoch_stop_cl"]

        for epoch in range(1, num_epoch + 1):
            self.do_online_sim()
            self.do_max_entropy_aug()

            do_instance_cl = (
                    (
                            (not use_warm_up4cluster_cl) or (use_warm_up4cluster_cl and epoch > epoch_warm_up4cluster_cl)
                    )
                    and (
                            not use_stop_cl_after or (epoch <= epoch_stop_cl)
                    )
            )
            if do_instance_cl:
                train_loader.dataset.set_use_aug()
            else:
                train_loader.dataset.set_not_use_aug()

            if instance_cl_config["use_weight_dynamic"]:
                weight_dynamic_type = instance_cl_config["weight_dynamic"]["type"]
                weight_dynamic_config = instance_cl_config["weight_dynamic"][weight_dynamic_type]
                if weight_dynamic_type == "multi_step":
                    for e, v in weight_dynamic_config["step_weight"]:
                        if epoch <= e:
                            weight_cl_loss = v
                            break
                elif weight_dynamic_type == "linear_increase":
                    e = weight_dynamic_config["epoch"]
                    v = weight_dynamic_config["value"]
                    weight_cl_loss = self.params["loss_config"]["cl loss"] + (epoch // e) * v
                else:
                    raise NotImplementedError()

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                num_seq = batch["mask_seq"].shape[0]
                loss = 0.

                if do_instance_cl:
                    if latent_type4cl in ["mean_pool", "last_time"]:
                        cl_loss = model.get_instance_cl_loss(batch, instance_cl_config, self.dataset_adv_generated)
                    elif latent_type4cl == "all_time":
                        cl_loss = model.get_instance_cl_loss_all_interaction(batch, instance_cl_config, self.dataset_adv_generated)
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
        use_stop_cl_after = self.params["other"]["instance_cl"]["use_stop_cl_after"]
        epoch_stop_cl = self.params["other"]["instance_cl"]["epoch_stop_cl"]
        do_instance_cl = not use_stop_cl_after or (current_epoch < epoch_stop_cl)

        model = self.objects["models"]["kt_model"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        if do_instance_cl and aug_type == "informative_aug" and use_online_sim and (not use_warm_up4online_sim or after_warm_up):
            t_start = get_now_time()
            concept_emb = model.get_concept_emb()
            train_loader.dataset.online_similarity.analysis(concept_emb)
            t_end = get_now_time()
            print(f"online similarity analysis: from {t_start} to {t_end}")

    def do_max_entropy_aug(self):
        use_adv_aug = self.params["other"]["instance_cl"]["use_adv_aug"]
        max_entropy_adv_aug_config = self.params["other"]["max_entropy_adv_aug"]
        current_epoch = self.train_record.get_current_epoch()
        epoch_interval_generate = max_entropy_adv_aug_config["epoch_interval_generate"]
        loop_adv = max_entropy_adv_aug_config["loop_adv"]
        epoch_generate = max_entropy_adv_aug_config["epoch_generate"]
        adv_learning_rate = max_entropy_adv_aug_config["adv_learning_rate"]
        eta = max_entropy_adv_aug_config["eta"]
        gamma = max_entropy_adv_aug_config["gamma"]

        do_generate = (current_epoch % epoch_interval_generate == 0) and (self.num_epoch_adv_gen < epoch_generate)
        model = self.objects["models"]["kt_model"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        if use_adv_aug and do_generate:
            t_start = get_now_time()
            model.eval()
            # RNN就需要加上torch.backends.cudnn.enabled = False，才能在eval模式下通过网络还能保留梯度
            # torch.backends.cudnn.enabled = False
            optimizer = self.init_data_generated(adv_learning_rate)
            for batch_idx, batch in enumerate(train_loader):
                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                adv_predict_loss, adv_entropy, adv_mse_loss = model.max_entropy_adv_aug(
                    self.dataset_adv_generated, batch, optimizer, loop_adv, eta, gamma
                )
                self.adv_loss.add_loss("gen pred loss", adv_predict_loss * num_sample, num_sample)
                self.adv_loss.add_loss("gen entropy loss", adv_entropy * num_sample, num_sample)
                self.adv_loss.add_loss("gen mse loss", adv_mse_loss * num_sample, num_sample)

            # torch.backends.cudnn.enabled = True
            self.num_epoch_adv_gen += 1
            t_end = get_now_time()
            print(f"max entropy adversarial data augment: from {t_start} to {t_end}, {self.adv_loss.get_str()}")
            self.adv_loss.clear_loss()
            self.data_generated_remove_grad()

    def init_data_generated(self, adv_learning_rate):
        model = self.objects["models"]["kt_model"]
        model_name = model.model_name

        self.dataset_adv_generated = {}
        if model_name == "qDKT":
            self.dataset_adv_generated["embed_layer"] = (
                KTEmbedLayer(self.params, self.objects).to(self.params["device"]))
            optimizer = optim.SGD(self.dataset_adv_generated["embed_layer"].parameters(), lr=adv_learning_rate)
        elif model_name == "AKT":
            encoder_layer_type = self.params["models_config"]["kt_model"]["encoder_layer"]["type"]
            encoder_layer_config = self.params["models_config"]["kt_model"]["encoder_layer"][encoder_layer_type]
            num_concept = encoder_layer_config["num_concept"]
            num_question = encoder_layer_config["num_question"]
            encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
            dim_emb = encoder_config["dim_model"]
            separate_qa = encoder_config["separate_qa"]
            self.dataset_adv_generated["embed_question_difficulty"] = (
                nn.Embedding(num_question, 1,
                             _weight=model.embed_question_difficulty.weight.detach().clone())
            )
            self.dataset_adv_generated["embed_concept_variation"] = (
                nn.Embedding(num_concept, dim_emb,
                             _weight=model.embed_concept_variation.weight.detach().clone())
            )
            self.dataset_adv_generated["embed_interaction_variation"] = (
                nn.Embedding(2 * num_concept, dim_emb,
                             _weight=model.embed_interaction_variation.weight.detach().clone())
            )
            self.dataset_adv_generated["embed_concept"] = (
                nn.Embedding(num_concept, dim_emb,
                             _weight=model.embed_concept.weight.detach().clone())
            )
            if separate_qa:
                self.dataset_adv_generated["embed_interaction"] = (
                    nn.Embedding(2 * num_concept + 1, dim_emb,
                                 _weight=model.embed_interaction.weight.detach().clone())
                )
            else:
                self.dataset_adv_generated["embed_interaction"] = (
                    nn.Embedding(2, dim_emb,
                                 _weight=model.embed_interaction.weight.detach().clone())
                )
            optimizer = optim.SGD(params=[
                self.dataset_adv_generated["embed_question_difficulty"].weight,
                self.dataset_adv_generated["embed_concept_variation"].weight,
                self.dataset_adv_generated["embed_interaction_variation"].weight,
                self.dataset_adv_generated["embed_concept"].weight,
                self.dataset_adv_generated["embed_interaction"].weight
            ], lr=adv_learning_rate)
        else:
            raise NotImplementedError()

        return optimizer

    def data_generated_remove_grad(self):
        model = self.objects["models"]["kt_model"]
        model_name = model.model_name

        if model_name == "qDKT":
            self.dataset_adv_generated["embed_layer"].embed_concept.weight.requires_grad_(False)
            self.dataset_adv_generated["embed_layer"].embed_question.weight.requires_grad_(False)
        elif model_name == "AKT":
            self.dataset_adv_generated["embed_question_difficulty"].weight.requires_grad_(False)
            self.dataset_adv_generated["embed_concept_variation"].weight.requires_grad_(False)
            self.dataset_adv_generated["embed_interaction_variation"].weight.requires_grad_(False)
            self.dataset_adv_generated["embed_concept"].weight.requires_grad_(False)
            self.dataset_adv_generated["embed_interaction"].weight.requires_grad_(False)
        else:
            raise NotImplementedError()
