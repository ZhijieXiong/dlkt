import torch
import torch.nn as nn
import torch.optim as optim

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .LossRecord import LossRecord
from ..model.Module.KTEmbedLayer import KTEmbedLayer
from ..util.basic import *


class MetaOptimizeCLTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(MetaOptimizeCLTrainer, self).__init__(params, objects)
        self.dataset_adv_generated = None
        self.num_epoch_adv_gen = 0
        self.adv_loss = LossRecord(["gen pred loss", "gen entropy loss", "gen mse loss"])

    def init_loss_record(self):
        used_losses = ["predict loss"]
        loss_config = self.params["loss_config"]
        for loss_name in loss_config:
            if loss_name in ["cl loss1", "cl loss2", "reg loss"]:
                used_losses.append(loss_name + " stage1")
                used_losses.append(loss_name + " stage2")
            else:
                used_losses.append(loss_name)
        return LossRecord(used_losses)

    def train(self):
        train_strategy = self.params["train_strategy"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        # kt model
        kt_model = self.objects["models"]["kt_model"]
        use_grad_clip = self.params["grad_clip_config"]["kt_model"]["use_clip"]
        grad_clipped = self.params["grad_clip_config"]["kt_model"].get("grad_clipped", 10)
        use_lr_scheduler = self.params["schedulers_config"]["kt_model"]["use_scheduler"]
        kt_optimizer = self.objects["optimizers"]["kt_model"]
        kt_scheduler = self.objects["schedulers"]["kt_model"]

        # extractor
        extractor0_model = self.objects["models"]["extractor0"]
        extractor1_model = self.objects["models"]["extractor1"]
        meta_extractors = (extractor0_model, extractor1_model)
        extractor0_optimizer = self.objects["optimizers"]["extractor0"]
        extractor1_optimizer = self.objects["optimizers"]["extractor1"]
        extractor0_scheduler = self.objects["schedulers"]["extractor0"]
        extractor1_scheduler = self.objects["schedulers"]["extractor1"]

        cl_type = self.params["other"]["instance_cl"]["cl_type"]
        use_adv_aug = self.params["other"]["max_entropy_aug"]["use_adv_aug"]
        use_regularization = self.params["other"]["meta_cl"]["use_regularization"]

        self.print_data_statics()

        weight_lambda = self.params["loss_config"]["cl loss1"]
        weight_beta = self.params["loss_config"]["cl loss2"]
        weight_gamma = self.params["loss_config"]["reg loss"]

        for epoch in range(1, num_epoch + 1):
            self.do_online_sim()
            self.do_max_entropy_aug()

            # 有对抗样本后，随机增强只需要生成一个view
            if use_adv_aug:
                dataset_config_this = self.params["datasets_config"]["train"]
                dataset_config_this["kt4aug"]["num_aug"] = 1

            kt_model.train()
            for batch_idx, batch in enumerate(train_loader):
                # step 1, update the parameters of the encoder，计算预测损失和随机数据增强的对比损失
                kt_optimizer.zero_grad()
                for param in extractor0_model.parameters():
                    param.requires_grad = False
                for param in extractor1_model.parameters():
                    param.requires_grad = False
                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                num_seq = batch["mask_seq"].shape[0]
                loss = 0.
                predict_loss = kt_model.get_predict_loss(batch)
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
                loss = loss + predict_loss
                if cl_type in ["mean_pool", "last_time"] and not use_adv_aug:
                    cl_loss1, cl_loss2, reg_loss = kt_model.meta_contrast(
                        batch, cl_type, meta_extractors, use_regularization
                    )
                elif cl_type in ["mean_pool", "last_time"] and use_adv_aug:
                    cl_loss1, cl_loss2, reg_loss = kt_model.meta_contrast_use_adv_aug(
                        batch, self.dataset_adv_generated, cl_type, meta_extractors, use_regularization
                    )
                else:
                    raise NotImplementedError()
                self.loss_record.add_loss("cl loss1 stage1",
                                          cl_loss1.detach().cpu().item() * num_seq, num_seq)
                self.loss_record.add_loss("cl loss2 stage1",
                                          cl_loss2.detach().cpu().item() * num_seq, num_seq)
                self.loss_record.add_loss("reg loss stage1",
                                          reg_loss.detach().cpu().item(), 1)
                loss = loss + weight_lambda * cl_loss1 + weight_beta * cl_loss2 + weight_gamma * reg_loss
                loss.backward()
                if use_grad_clip:
                    nn.utils.clip_grad_norm_(kt_model.parameters(), max_norm=grad_clipped)
                kt_optimizer.step()

                # step 2，update the parameters of two learnable extractors
                loss = 0.
                extractor0_optimizer.zero_grad()
                extractor1_optimizer.zero_grad()
                for param in extractor0_model.parameters():
                    param.requires_grad = True
                for param in extractor1_model.parameters():
                    param.requires_grad = True
                if cl_type in ["mean_pool", "last_time"] and not use_adv_aug:
                    cl_loss1, cl_loss2, reg_loss = kt_model.meta_contrast(
                        batch, cl_type, meta_extractors, use_regularization
                    )
                elif cl_type in ["mean_pool", "last_time"] and use_adv_aug:
                    cl_loss1, cl_loss2, reg_loss = kt_model.meta_contrast_use_adv_aug(
                        batch, self.dataset_adv_generated, cl_type, meta_extractors, use_regularization
                    )
                else:
                    raise NotImplementedError()
                self.loss_record.add_loss("cl loss1 stage2",
                                          cl_loss1.detach().cpu().item() * num_seq, num_seq)
                self.loss_record.add_loss("cl loss2 stage2",
                                          cl_loss2.detach().cpu().item() * num_seq, num_seq)
                self.loss_record.add_loss("reg loss stage2",
                                          reg_loss.detach().cpu().item(), 1)
                loss = loss + weight_lambda * cl_loss1 + cl_loss2 + weight_gamma * reg_loss
                loss.backward()
                if use_grad_clip:
                    nn.utils.clip_grad_norm_(extractor0_model.parameters(), max_norm=grad_clipped)
                    nn.utils.clip_grad_norm_(extractor1_model.parameters(), max_norm=grad_clipped)
                extractor0_optimizer.step()
                extractor1_optimizer.step()

            if use_lr_scheduler:
                kt_scheduler.step()
                extractor0_scheduler.step()
                extractor1_scheduler.step()
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
            concept_emb = model.get_concept_emb()
            train_loader.dataset.online_similarity.analysis(concept_emb)
            t_end = get_now_time()
            print(f"online similarity analysis: from {t_start} to {t_end}")

    def do_max_entropy_aug(self):
        use_adv_aug = self.params["other"]["max_entropy_aug"]["instance cl"]["use_adv_aug"]
        max_entropy_adv_aug_config = self.params["other"]["max_entropy_aug"]
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
            torch.backends.cudnn.enabled = False
            optimizer = self.init_data_generated(adv_learning_rate)
            for batch_idx, batch in enumerate(train_loader):
                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                adv_predict_loss, adv_entropy, adv_mse_loss = model.max_entropy_adv_aug(
                        self.dataset_adv_generated, batch, optimizer, loop_adv, eta, gamma
                )
                self.adv_loss.add_loss("gen pred loss", adv_predict_loss * num_sample, num_sample)
                self.adv_loss.add_loss("gen entropy loss", adv_entropy * num_sample, num_sample)
                self.adv_loss.add_loss("gen mse loss", adv_mse_loss * num_sample, num_sample)

            torch.backends.cudnn.enabled = True
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
