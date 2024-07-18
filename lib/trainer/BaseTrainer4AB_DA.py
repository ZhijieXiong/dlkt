import torch
import torch.nn as nn
import torch.optim as optim

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .LossRecord import LossRecord
from ..model.Module.KTEmbedLayer import KTEmbedLayer
from ..util.basic import get_now_time
from ..evaluator.util import get_seq_fine_grained_sample_mask, get_question_fine_grained_sample_mask


class BaseTrainer4AB_DA(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(BaseTrainer4AB_DA, self).__init__(params, objects)
        self.dataset_adv_generated = None
        self.adv_loss = LossRecord(["ada gen pred loss", "ada gen mse loss", "ada gen entropy"])

    def do_adv_aug(self):
        adv_bias_aug = self.params["other"]["adv_bias_aug"]
        current_epoch = self.train_record.get_current_epoch()
        epoch_interval_generate = adv_bias_aug["epoch_interval_generate"]
        loop_adv = adv_bias_aug["loop_adv"]
        adv_learning_rate = adv_bias_aug["adv_learning_rate"]
        gamma = adv_bias_aug["gamma"]
        eta = adv_bias_aug["eta"]
        ablation = adv_bias_aug["ablation"]

        do_generate = (current_epoch % epoch_interval_generate == 0)
        model = self.objects["models"]["kt_model"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        if do_generate:
            t_start = get_now_time()
            model.eval()
            # RNN就需要加上torch.backends.cudnn.enabled = False，才能在eval模式下通过网络还能保留梯度，否则报错：RuntimeError: cudnn RNN backward can only be called in training mode
            # 不使用RNN就可以不加
            use_rnn = model.model_name in ["qDKT"]
            if use_rnn:
                torch.backends.cudnn.enabled = False

            optimizer = self.init_data_generated(adv_learning_rate)
            for batch_idx, batch in enumerate(train_loader):
                batch_ = {
                    "question_seqs": batch["question_seq"][:, 1:].detach().tolist(),
                    "label_seqs": batch["correct_seq"][:, 1:].detach().tolist(),
                    "mask_seqs": batch["mask_seq"][:, 1:].detach().tolist()
                }
                seq_easy_mask, seq_normal_mask, seq_hard_mask = get_seq_fine_grained_sample_mask(batch_, 10, 0.4)
                train_statics_common = self.objects["data"].get("train_data_statics_common", None)
                question_easy_mask, question_normal_mask, question_hard_mask = \
                    get_question_fine_grained_sample_mask(batch_, train_statics_common, 0.4)

                if ablation in [0, 3]:
                    bias_aligned_mask = torch.BoolTensor(question_easy_mask).to(self.params["device"])
                    bias_conflicting_mask = torch.BoolTensor(question_hard_mask).to(self.params["device"])
                elif ablation in [1, 4]:
                    bias_aligned_mask = torch.BoolTensor(seq_easy_mask).to(self.params["device"])
                    bias_conflicting_mask = torch.BoolTensor(seq_hard_mask).to(self.params["device"])
                elif ablation in [2, 5, 6, 7, 8, 9]:
                    bias_aligned_mask = torch.BoolTensor(question_easy_mask).to(self.params["device"]) | \
                                        torch.BoolTensor(seq_easy_mask).to(self.params["device"])
                    bias_conflicting_mask = torch.BoolTensor(question_hard_mask).to(self.params["device"]) | \
                                            torch.BoolTensor(seq_hard_mask).to(self.params["device"])
                else:
                    raise NotImplementedError()

                num_sample = torch.sum(bias_aligned_mask).item()
                if ablation in [0, 1, 2, 3, 4, 5]:
                    adv_predict_loss, adv_mse_loss, adv_entropy = model.adv_bias_aug(
                        self.dataset_adv_generated, batch, optimizer, loop_adv, eta, gamma, bias_aligned_mask
                    )
                elif ablation in [6, 7]:
                    adv_predict_loss, adv_mse_loss, adv_entropy = model.adv_bias_aug(
                        self.dataset_adv_generated, batch, optimizer, loop_adv, eta, gamma, bias_conflicting_mask
                    )
                elif ablation in [8, 9]:
                    adv_predict_loss, adv_mse_loss, adv_entropy = model.adv_bias_aug(
                        self.dataset_adv_generated, batch, optimizer, loop_adv, eta, gamma
                    )
                else:
                    raise NotImplementedError()
                self.adv_loss.add_loss("ada gen pred loss", adv_predict_loss * num_sample, num_sample)
                self.adv_loss.add_loss("ada gen mse loss", adv_mse_loss * num_sample, num_sample)
                self.adv_loss.add_loss("ada gen entropy", adv_entropy * num_sample, num_sample)

            if use_rnn:
                torch.backends.cudnn.enabled = True

            t_end = get_now_time()
            self.objects["logger"].info(
                f"max entropy adversarial data augment: from {t_start} to {t_end}, {self.adv_loss.get_str()}")
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
            encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
            num_concept = encoder_config["num_concept"]
            num_question = encoder_config["num_question"]
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
        elif model_name == "DIMKT":
            encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
            dim_emb = encoder_config["dim_emb"]
            num_question = encoder_config["num_question"]
            num_concept = encoder_config["num_concept"]
            num_question_diff = encoder_config["num_question_diff"]
            num_concept_diff = encoder_config["num_concept_diff"]

            self.dataset_adv_generated["embed_question"] = \
                nn.Embedding(num_question, dim_emb,
                             _weight=model.embed_question.weight.detach().clone())
            self.dataset_adv_generated["embed_concept"] = \
                nn.Embedding(num_concept, dim_emb,
                             _weight=model.embed_concept.weight.detach().clone())
            self.dataset_adv_generated["embed_question_diff"] = \
                nn.Embedding(num_question_diff + 1, dim_emb,
                             _weight=model.embed_question_diff.weight.detach().clone())
            self.dataset_adv_generated["embed_concept_diff"] = \
                nn.Embedding(num_concept_diff + 1, dim_emb,
                             _weight=model.embed_concept_diff.weight.detach().clone())
            self.dataset_adv_generated["embed_correct"] = \
                nn.Embedding(2, dim_emb,
                             _weight=model.embed_correct.weight.detach().clone())
            optimizer = optim.SGD(params=[
                self.dataset_adv_generated["embed_question"].weight,
                self.dataset_adv_generated["embed_concept"].weight,
                self.dataset_adv_generated["embed_question_diff"].weight,
                self.dataset_adv_generated["embed_concept_diff"].weight,
                self.dataset_adv_generated["embed_correct"].weight
            ], lr=adv_learning_rate)
        elif model_name == "LPKT":
            encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT"]
            num_question = encoder_config["num_question"]
            dim_k = encoder_config["dim_k"]

            self.dataset_adv_generated["e_embed"] = (
                nn.Embedding(num_question + 1, dim_k, _weight=model.e_embed.weight.detach().clone())
            )
            self.dataset_adv_generated["at_embed"] = (
                nn.Embedding(3600 + 1, dim_k, _weight=model.at_embed.weight.detach().clone())
            )
            self.dataset_adv_generated["it_embed"] = (
                nn.Embedding(43200 + 1, dim_k, _weight=model.it_embed.weight.detach().clone())
            )
            optimizer = optim.SGD(params=[
                self.dataset_adv_generated["e_embed"].weight,
                self.dataset_adv_generated["at_embed"].weight,
                self.dataset_adv_generated["it_embed"].weight
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
        elif model_name == "DIMKT":
            self.dataset_adv_generated["embed_question"].weight.requires_grad_(False),
            self.dataset_adv_generated["embed_concept"].weight.requires_grad_(False),
            self.dataset_adv_generated["embed_question_diff"].weight.requires_grad_(False),
            self.dataset_adv_generated["embed_concept_diff"].weight.requires_grad_(False),
            self.dataset_adv_generated["embed_correct"].weight.requires_grad_(False)
        elif model_name == "LPKT":
            self.dataset_adv_generated["e_embed"].weight.requires_grad_(False),
            self.dataset_adv_generated["at_embed"].weight.requires_grad_(False),
            self.dataset_adv_generated["it_embed"].weight.requires_grad_(False)
        else:
            raise NotImplementedError()
