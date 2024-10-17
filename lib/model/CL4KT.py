import torch

from torch.nn import Module, Embedding, Linear, Dropout, ModuleList, Sequential, CosineSimilarity
from torch.nn.functional import binary_cross_entropy, cross_entropy
from torch.nn.modules.activation import GELU
from .Module.CL4KT_Transformer import CL4KTTransformerLayer


class Similarity(Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

    def set_temp(self, temp):
        self.temp = temp


# 加上这句的话dataloader需要加上 generator=torch.Generator(self.params["device='cuda')
# if torch.cuda.is_available():
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)


class CL4KT(Module):
    model_name = "CL4KT"
    use_question = False

    def __init__(self, params, objects):
        super(CL4KT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["CL4KT"]
        num_concept = encoder_config["num_concept"]
        dim_model = encoder_config["dim_model"]
        num_block = encoder_config["num_block"]
        dim_final_fc = encoder_config["dim_final_fc"]
        dropout = encoder_config["dropout"]
        temp = params["other"]["instance_cl"]["temp"]

        self.embed_concept = Embedding(num_concept, dim_model, padding_idx=0)
        self.embed_interaction = Embedding(2 * num_concept, dim_model, padding_idx=0)
        self.sim = Similarity(temp=temp)

        self.concept_encoder = ModuleList([CL4KTTransformerLayer(params, objects) for _ in range(num_block)])
        self.interaction_encoder = ModuleList([CL4KTTransformerLayer(params, objects) for _ in range(num_block)])
        self.knowledge_retriever = ModuleList([CL4KTTransformerLayer(params, objects) for _ in range(num_block)])

        self.predict_layer = Sequential(
            Linear(2 * dim_model, dim_final_fc),
            GELU(),
            Dropout(dropout),
            Linear(dim_final_fc, dim_final_fc // 2),
            GELU(),
            Dropout(dropout),
            Linear(dim_final_fc // 2, 1),
        )

    def get_concept_emb(self):
        return self.embed_concept.weight

    def get_interaction_emb(self, concept_seq, correct_seq):
        num_concept = self.params["models_config"]["kt_model"]["encoder_layer"]["CL4KT"]["num_concept"]
        interaction_seq = concept_seq + num_concept * correct_seq
        return self.embed_interaction(interaction_seq)

    def forward_backbone(self, concept_seq, correct_seq):
        concept_emb = self.embed_concept(concept_seq)
        interaction_emb = self.get_interaction_emb(concept_seq, correct_seq)

        x, y = concept_emb, interaction_emb

        # CL4KT原始代码写法
        # self attention，即分别对question和interaction编码时，peek current
        for block in self.concept_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=True)

        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, apply_pos=True)

        # cross attention，即对question和interaction编码时，dont peek current response
        for block in self.knowledge_retriever:
            x, _ = block(mask=0, query=x, key=x, values=y, apply_pos=True)

        return x, concept_emb

    def get_concept_seq_state(self, concept_seq, mask_seq):
        concept_emb = self.embed_concept(concept_seq)
        x = concept_emb
        for block in self.concept_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=False)
        x_final = (x * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return x_final

    def get_state(self, concept_seq, correct_seq, mask_seq):
        interaction_emb = self.get_interaction_emb(concept_seq, correct_seq)
        y = interaction_emb
        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, apply_pos=False)
        y_final = (y * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return y_final

    def forward(self, concept_seq, correct_seq):
        encoder_out, concept_emb = self.forward_backbone(concept_seq, correct_seq)
        retrieved_knowledge = torch.cat([encoder_out, concept_emb], dim=-1)
        model_output = torch.sigmoid(self.predict_layer(retrieved_knowledge)).squeeze(dim=-1)

        return model_output

    def get_predict_loss(self, batch):
        weight_cl_loss = self.params["loss_config"]["cl loss"]
        concept_seq, correct_seq = batch["concept_seq"], batch["correct_seq"]

        loss = 0.
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(concept_seq, correct_seq)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch, batch["mask_seq"][:, 1:].bool())
        ground_truth = torch.masked_select(correct_seq[:, 1:].long(), mask_seq[:, 1:])
        predict_loss = binary_cross_entropy(predict_score.double(), ground_truth.double())
        loss = loss + predict_loss

        cl_loss = self.get_cl_loss(batch)
        loss = loss + cl_loss * weight_cl_loss

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        num_seq = batch["mask_seq"].shape[0]
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "cl loss": {
                    "value": cl_loss.detach().cpu().item() * num_seq,
                    "num_sample": num_seq
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_score(self, batch):
        concept_seq, correct_seq = batch["concept_seq"], batch["correct_seq"]
        predict_score_batch = self.forward(concept_seq, correct_seq)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch, batch["mask_seq"][:, 1:].bool())

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_cl_loss(self, batch):
        use_hard_neg = self.params["other"]["instance_cl"]["use_hard_neg"]
        hard_negative_weight = self.params["other"]["instance_cl"]["hard_neg_weight"]

        concept_seq_aug0 = batch["concept_seq_aug_0"]
        concept_seq_aug1 = batch["concept_seq_aug_1"]
        correct_seq_aug0 = batch["correct_seq_aug_0"]
        correct_seq_aug1 = batch["correct_seq_aug_1"]
        mask_seq_aug0 = batch["mask_seq_aug_0"]
        mask_seq_aug1 = batch["mask_seq_aug_1"]
        concept_score_aug1 = self.get_concept_seq_state(concept_seq_aug0, mask_seq_aug0)
        concept_score_aug2 = self.get_concept_seq_state(concept_seq_aug1, mask_seq_aug1)
        concept_cos_sim = self.sim(concept_score_aug1.unsqueeze(1), concept_score_aug2.unsqueeze(0))
        concept_labels = torch.arange(concept_cos_sim.size(0)).long().to(self.params["device"])
        concept_cl_loss = cross_entropy(concept_cos_sim, concept_labels)

        interaction_score_aug1 = self.get_state(concept_seq_aug0, correct_seq_aug0, mask_seq_aug0)
        interaction_score_aug2 = self.get_state(concept_seq_aug1, correct_seq_aug1, mask_seq_aug1)
        interaction_cos_sim = self.sim(interaction_score_aug1.unsqueeze(1), interaction_score_aug2.unsqueeze(0))

        if use_hard_neg:
            correct_seq_neg = batch["correct_seq_hard_neg"]
            interaction_score_neg = self.get_state(concept_seq_aug0, correct_seq_neg, mask_seq_aug0)
            interaction_negative_cos_sim = self.sim(interaction_score_aug1.unsqueeze(1), interaction_score_neg.unsqueeze(0))
            interaction_cos_sim = torch.cat([interaction_cos_sim, interaction_negative_cos_sim], 1)

            weights = torch.tensor(
                [
                    [0.0] * (interaction_cos_sim.size(-1) - interaction_negative_cos_sim.size(-1))
                    + [0.0] * i
                    + [hard_negative_weight]
                    + [0.0] * (interaction_negative_cos_sim.size(-1) - i - 1)
                    for i in range(interaction_negative_cos_sim.size(-1))
                ]
            ).to(self.params["device"])
            interaction_cos_sim = interaction_cos_sim + weights

        interaction_labels = torch.arange(interaction_cos_sim.size(0)).long().to(self.params["device"])
        interaction_cl_loss = cross_entropy(interaction_cos_sim, interaction_labels)

        cl_loss = concept_cl_loss + interaction_cl_loss

        return cl_loss
