import torch
from torch import nn

from .Module.SparseKT_Block import Architecture
from .Module.KTEmbedLayer import KTEmbedLayer


class SparseKT(nn.Module):
    model_name = "SparseKT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
        num_concept = encoder_config["num_concept"]
        num_question = encoder_config["num_question"]
        difficulty_scalar = encoder_config["difficulty_scalar"]
        dim_model = encoder_config["dim_model"]
        dim_final_fc = encoder_config["dim_final_fc"]
        dim_final_fc2 = encoder_config["dim_final_fc2"]
        separate_qa = encoder_config["separate_qa"]
        dropout = encoder_config["dropout"]

        if difficulty_scalar:
            self.embed_question_difficulty = nn.Embedding(num_question, 1)
        else:
            self.embed_question_difficulty = nn.Embedding(num_question, dim_model)
        self.embed_concept_variation = nn.Embedding(num_concept, dim_model)
        self.embed_interaction_variation = nn.Embedding(2 * num_concept, dim_model)

        self.embed_concept = nn.Embedding(num_concept, dim_model)
        if separate_qa:
            self.embed_interaction = nn.Embedding(2 * num_concept, dim_model)
        else:
            # false default
            self.embed_interaction = nn.Embedding(2, dim_model)

        self.model = Architecture(params)

        self.out = nn.Sequential(
            nn.Linear(dim_model + dim_model, dim_final_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc, dim_final_fc2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc2, 1),
        )

        self.reset()

    def reset(self):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
        num_question = encoder_config["num_question"]
        for p in self.parameters():
            if p.size(0) == num_question:
                torch.nn.init.constant_(p, 0.0)

    def get_concept_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                self.embed_concept,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_emb = self.embed_concept(batch["concept_seq"])

        return concept_emb

    def get_concept_variation_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                self.embed_concept_variation,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_emb = self.embed_concept_variation(batch["concept_seq"])

        return concept_emb

    def base_emb(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
        separate_qa = encoder_config["separate_qa"]
        correct_seq = batch["correct_seq"]

        # c_ct
        concept_emb = self.get_concept_emb(batch)
        if separate_qa:
            # todo: 有问题，如果是only question也要融合interaction_emb
            concept_seq = batch["concept_seq"]
            interaction_seq = concept_seq + self.num_concept * correct_seq
            interaction_emb = self.embed_interaction(interaction_seq)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_interaction(correct_seq) + concept_emb
        return concept_emb, interaction_emb

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
        use_akt_rasch = encoder_config["use_akt_rasch"]
        separate_qa = encoder_config["separate_qa"]

        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        concept_variation_emb = self.get_concept_variation_emb(batch)
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + concept_variation_emb * question_difficulty_emb

        if use_akt_rasch:
            interaction_variation_emb = self.embed_interaction_variation(correct_seq)
            if separate_qa:
                # uq * f_(ct,rt) + e_(ct,rt)
                interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
            else:
                # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
                interaction_emb = \
                    interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb)

        d_output = self.model(question_emb, interaction_emb)

        concat_q = torch.cat([d_output, question_emb], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        predict_score = m(output)

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        if self.params.get("use_sample_weight", False):
            weight = torch.masked_select(batch["weight_seq"][:, 1:], mask_bool_seq[:, 1:])
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(),
                                                              ground_truth.double(),
                                                              weight=weight)
        else:
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss

    def get_predict_loss_per_sample(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss_per_sample = nn.functional.binary_cross_entropy(
            predict_score.double(), ground_truth.double(), reduction="none"
        )

        return predict_loss_per_sample

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)[:, 1:]
