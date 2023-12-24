import torch
import numpy as np
import torch.nn as nn

from ..util.parse import concept2question_from_Q, question2concept_from_Q


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class NCD4KT(nn.Module):
    model_name = "NCD4KT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["NCD4KT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_concept = encoder_config["dim_concept"]
        dim_correct = encoder_config["dim_correct"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]

        self.embed_concept = nn.Embedding(num_concept, dim_concept)
        diff_init = torch.from_numpy(objects["data"]["Q_table"]).float().to(params["device"])
        diff_init[diff_init == 0] = -1
        self.embed_question_diff = nn.Embedding(num_question, num_concept, _weight=diff_init)
        disc_init = torch.ones((num_question, 1)).to(params["device"])
        self.embed_question_disc = nn.Embedding(num_question, 1, _weight=disc_init)

        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_concept + dim_correct, num_concept, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_concept + dim_correct, num_concept, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_concept + dim_correct, num_concept, batch_first=True, num_layers=num_rnn_layer)
        self.predict_layer = nn.Sequential(
            nn.Linear(num_concept, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        # 解析q table 用于多知识点embedding融合，如对于一道多知识点习题的知识点embedding取平均值作为该习题的embedding
        Q_table = objects["data"]["Q_table"]
        self.question2concept_list = question2concept_from_Q(Q_table)
        self.concept2question_list = concept2question_from_Q(Q_table)
        question2concept_table = []
        question2concept_mask_table = []
        num_max_c_in_q = np.max(np.sum(Q_table, axis=1))
        num_question = Q_table.shape[0]
        for i in range(num_question):
            cs = np.argwhere(Q_table[i] == 1).reshape(-1).tolist()
            pad_len = num_max_c_in_q - len(cs)
            question2concept_table.append(cs + [0] * pad_len)
            question2concept_mask_table.append([1] * len(cs) + [0] * pad_len)
        self.question2concept_table = torch.tensor(question2concept_table).long().to(params["device"])
        self.question2concept_mask_table = torch.tensor(question2concept_mask_table).long().to(params["device"])
        self.num_max_concept = num_max_c_in_q

    def get_concept_fusion_emb(self, batch):
        question_seq = batch["question_seq"]

        q2c_index = self.question2concept_table[question_seq]
        # bs * seq_len * num_max_c * dim
        concept_emb = self.embed_concept(q2c_index)
        question_diff_emb = self.embed_question_diff(question_seq)
        # bs * seq_len * num_max_c * dim
        question_diff_emb = torch.gather(question_diff_emb, 2, q2c_index).unsqueeze(-1)
        # bs * seq_len * num_max_c * dim
        concept_mask = self.question2concept_mask_table[question_seq]
        concept_fusion_emb = ((concept_emb * question_diff_emb) * concept_mask.unsqueeze(-1)).sum(-2)
        concept_fusion_emb = concept_fusion_emb / concept_mask.sum(-1).unsqueeze(-1)

        return concept_fusion_emb

    def forward(self, batch):
        dim_correct = self.params["models_config"]["kt_model"]["encoder_layer"]["NCD4KT"]["dim_correct"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        concept_fusion_emb = self.get_concept_fusion_emb(batch)
        interaction_emb = torch.cat((concept_fusion_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        latent = torch.sigmoid(latent)
        question_diff_emb = torch.sigmoid(self.embed_question_diff(question_seq[:, 1:]))
        question_disc_emb = torch.sigmoid(self.embed_question_disc(question_seq[:, 1:])) * 10
        q2c_index = self.question2concept_table[question_seq[:, 1:]]
        concept_relate = torch.zeros_like(latent).to(self.params["device"])
        concept_relate[
            torch.arange(batch_size).unsqueeze(1).unsqueeze(2),
            torch.arange(seq_len-1).unsqueeze(0).unsqueeze(2),
            q2c_index
        ] = 1
        predict_score = question_disc_emb * (latent - question_diff_emb) * concept_relate
        predict_score = torch.sigmoid(self.predict_layer(predict_score)).squeeze(dim=-1)

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.predict_layer.apply(clipper)
