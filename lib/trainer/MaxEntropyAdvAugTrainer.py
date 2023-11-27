import torch
import torch.nn as nn

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .LossRecord import LossRecord
from ..model.Module.KTEmbedLayer import KTEmbedLayer
from ..util.basic import *


class MaxEntropyAdvAugTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(MaxEntropyAdvAugTrainer, self).__init__(params, objects)
        self.dataset_adv_generated = None
        self.num_epoch_adv_gen = 0
        self.adv_loss = LossRecord(["gen pred loss", "gen entropy loss", "gen mse loss"])

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

        train_statics = train_loader.dataset.get_statics_kt_dataset()
        print(f"train, seq: {train_statics[0]}, sample: {train_statics[1]}, accuracy: {train_statics[2]:<.4}")
        if train_strategy["type"] == "valid_test":
            valid_statics = self.objects["data_loaders"]["valid_loader"].dataset.get_statics_kt_dataset()
            print(f"valid, seq: {valid_statics[0]}, sample: {valid_statics[1]}, accuracy: {valid_statics[2]:<.4}")
        test_statics = test_loader.dataset.get_statics_kt_dataset()
        print(f"test, seq: {test_statics[0]}, sample: {test_statics[1]}, accuracy: {test_statics[2]:<.4}")

        use_warm_up = self.params["other"]["max_entropy_adv_aug"]["use_warm_up"]
        epoch_warm_up = self.params["other"]["max_entropy_adv_aug"]["epoch_warm_up"]
        weight_adv_pred_loss = self.params["loss_config"]["adv predict loss"]
        for epoch in range(1, num_epoch + 1):
            self.do_max_entropy_aug_()

            use_adv_aug = not use_warm_up or (epoch > epoch_warm_up)
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                predict_loss = model.get_predict_loss(batch)
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
                predict_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

                if use_adv_aug:
                    optimizer.zero_grad()
                    adv_aug_predict_loss = model.get_predict_loss_from_adv_data(self.dataset_adv_generated, batch)
                    self.loss_record.add_loss("adv predict loss",
                                              adv_aug_predict_loss.detach().cpu().item() * num_sample, num_sample)
                    adv_aug_predict_loss = weight_adv_pred_loss * adv_aug_predict_loss
                    adv_aug_predict_loss.backward()
                    if grad_clip_config["use_clip"]:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                    optimizer.step()

            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break

    def do_max_entropy_aug(self):
        max_entropy_adv_aug_config = self.params["other"]["max_entropy_adv_aug"]
        use_warm_up = max_entropy_adv_aug_config["use_warm_up"]
        epoch_warm_up = max_entropy_adv_aug_config["epoch_warm_up"]
        current_epoch = self.train_record.get_current_epoch()
        epoch_interval_generate = max_entropy_adv_aug_config["epoch_interval_generate"]
        loop_adv = max_entropy_adv_aug_config["loop_adv"]
        epoch_generate = max_entropy_adv_aug_config["epoch_generate"]
        adv_learning_rate = max_entropy_adv_aug_config["adv_learning_rate"]
        eta = max_entropy_adv_aug_config["eta"]
        gamma = max_entropy_adv_aug_config["gamma"]

        do_generate = ((current_epoch - epoch_warm_up) % epoch_interval_generate == 0)
        do_generate = ((not use_warm_up) or
                       (do_generate and (self.num_epoch_adv_gen < epoch_generate) and (current_epoch >= epoch_warm_up)))
        model = self.objects["models"]["kt_model"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        if do_generate:
            model.eval()
            # RNN就需要加上torch.backends.cudnn.enabled = False，才能在eval模式下通过网络还能保留梯度
            torch.backends.cudnn.enabled = False

            data_generated = {
                "seq_id": [],
                "emb_seq": []
            }
            for batch_idx, batch in enumerate(train_loader):
                num_seq = batch["mask_seq"].shape[0]
                inputs_max, adv_predict_loss, adv_entropy, adv_mse_loss = (
                    model.get_max_entropy_adv_aug_emb(batch, adv_learning_rate, loop_adv, eta, gamma))
                self.adv_loss.add_loss("gen pred loss", adv_predict_loss * num_seq, num_seq)
                self.adv_loss.add_loss("gen entropy loss", adv_entropy * num_seq, num_seq)
                self.adv_loss.add_loss("gen mse loss", adv_mse_loss * num_seq, num_seq)
                data_generated["seq_id"].append(batch["seq_id"].to("cpu"))
                data_generated["emb_seq"].append(inputs_max.detach().clone().to("cpu"))

            print(self.adv_loss.get_str())
            self.adv_loss.clear_loss()
            for k in data_generated:
                data_generated[k] = torch.cat(data_generated[k], dim=0)
            self.save_adv_data(data_generated)

            torch.backends.cudnn.enabled = True
            self.num_epoch_adv_gen += 1

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

    def do_max_entropy_aug_(self):
        max_entropy_adv_aug_config = self.params["other"]["max_entropy_adv_aug"]
        use_warm_up = max_entropy_adv_aug_config["use_warm_up"]
        epoch_warm_up = max_entropy_adv_aug_config["epoch_warm_up"]
        current_epoch = self.train_record.get_current_epoch()
        epoch_interval_generate = max_entropy_adv_aug_config["epoch_interval_generate"]
        loop_adv = max_entropy_adv_aug_config["loop_adv"]
        epoch_generate = max_entropy_adv_aug_config["epoch_generate"]
        adv_learning_rate = max_entropy_adv_aug_config["adv_learning_rate"]
        eta = max_entropy_adv_aug_config["eta"]
        gamma = max_entropy_adv_aug_config["gamma"]

        do_generate = ((current_epoch - epoch_warm_up) % epoch_interval_generate == 0)
        do_generate = ((not use_warm_up) or
                       (do_generate and (self.num_epoch_adv_gen < epoch_generate) and (current_epoch >= epoch_warm_up)))
        model = self.objects["models"]["kt_model"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        if do_generate:
            t_start = get_now_time()
            model.eval()
            # RNN就需要加上torch.backends.cudnn.enabled = False，才能在eval模式下通过网络还能保留梯度
            torch.backends.cudnn.enabled = False
            self.init_data_generated()
            for batch_idx, batch in enumerate(train_loader):
                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                adv_predict_loss, adv_entropy, adv_mse_loss = (
                    model.max_entropy_adv_aug(
                        self.dataset_adv_generated, batch, adv_learning_rate, loop_adv, eta, gamma
                    )
                )
                self.adv_loss.add_loss("gen pred loss", adv_predict_loss * num_sample, num_sample)
                self.adv_loss.add_loss("gen entropy loss", adv_entropy * num_sample, num_sample)
                self.adv_loss.add_loss("gen mse loss", adv_mse_loss * num_sample, num_sample)

            torch.backends.cudnn.enabled = True
            self.num_epoch_adv_gen += 1
            t_end = get_now_time()
            print(f"max entropy adversarial data augment: from {t_start} to {t_end}, {self.adv_loss.get_str()}")
            self.adv_loss.clear_loss()

    def init_data_generated(self):
        model = self.objects["models"]["kt_model"]
        model_name = model.model_name

        self.dataset_adv_generated = {}
        if model_name == "qDKT":
            self.dataset_adv_generated["embed_layer"] = (
                KTEmbedLayer(self.params, self.objects).to(self.params["device"]))
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
        else:
            raise NotImplementedError()
