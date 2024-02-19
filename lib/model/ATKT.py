import torch
import torch.nn as nn

from torch.nn.functional import one_hot
from torch.autograd import Variable, grad

from .util import l2_normalize_adv


class ATKT(nn.Module):
    model_name = "ATKT"

    def __init__(self, params, objects):
        super(ATKT, self).__init__()
        self.params = params
        self.objects = objects

        # embed layer
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["ATKT"]
        use_concept = encoder_config["use_concept"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_concept = encoder_config["dim_concept"]
        dim_latent = encoder_config["dim_latent"]
        dim_attention = encoder_config["dim_attention"]
        dim_correct = encoder_config["dim_correct"]
        dropout = encoder_config["dropout"]

        self.rnn = nn.LSTM(dim_concept + dim_correct, dim_latent, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_latent * 2, num_concept)
        self.sig = nn.Sigmoid()

        if use_concept:
            self.embed_concept = nn.Embedding(num_concept, dim_concept)
        else:
            self.embed_concept = nn.Embedding(num_question, dim_concept)
        self.embed_concept.weight.data[-1] = 0

        self.embed_correct = nn.Embedding(2, dim_correct)
        self.embed_correct.weight.data[-1] = 0

        self.dim_attention = dim_attention
        self.mlp = nn.Linear(dim_latent, dim_attention)
        self.similarity = nn.Linear(dim_attention, 1, bias=False)

    def attention_module(self, lstm_output):
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)

        # 这一步导致数据泄露！！！计算softmax时没有屏蔽未来的数据
        # (bs, seq_len, 1) -> (bs, seq_len, 1)
        # alphas = nn.Softmax(dim=1)(att_w)
        # attn_output = att_w * lstm_output

        # pykt修改后的代码
        seq_len = lstm_output.shape[1]
        attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(self.params["device"])
        att_w = att_w.transpose(1, 2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
        att_w = att_w.masked_fill_(attn_mask, float("-inf"))
        alphas = torch.nn.functional.softmax(att_w, dim=-1)
        attn_output = torch.bmm(alphas, lstm_output)

        attn_output_cum = torch.cumsum(attn_output, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_output
        final_output = torch.cat((attn_output_cum_1, lstm_output), 2)

        return final_output

    def get_interaction_emb(self, batch):
        use_concept = self.params["models_config"]["kt_model"]["encoder_layer"]["ATKT"]["use_concept"]
        if use_concept:
            concept_seq = batch["concept_seq"]
        else:
            concept_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        concept_emb = self.embed_concept(concept_seq)
        correct_emb = self.embed_correct(correct_seq)
        concept_correct_emb = torch.cat((concept_emb, correct_emb), 2)
        correct_concept_emb = torch.cat((correct_emb, concept_emb), 2)
        correct_seq = correct_seq.unsqueeze(2).expand_as(concept_correct_emb)
        interaction_emb = torch.where(correct_seq == 1, concept_correct_emb, correct_concept_emb)

        return interaction_emb

    def forward(self, batch):
        interaction_emb = self.get_interaction_emb(batch)
        predict_score = self.forward_from_emb(interaction_emb)

        return predict_score

    def forward_from_emb(self, interaction_emb):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(interaction_emb)
        latent = self.attention_module(rnn_out)
        predict_score = self.sig(self.fc(self.dropout_layer(latent)))

        return predict_score

    def get_predict_score(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["ATKT"]
        use_concept = encoder_config["use_concept"]
        if use_concept:
            num_concept = encoder_config["num_concept"]
        else:
            num_concept = encoder_config["num_question"]
        predict_score = self.forward(batch)[:, :-1]
        concept_seq = batch["concept_seq"]
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = (predict_score * one_hot(concept_seq[:, 1:].long(), num_concept)).sum(-1)
        predict_score = torch.masked_select(predict_score, mask_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["ATKT"]
        use_concept = encoder_config["use_concept"]
        if use_concept:
            num_concept = encoder_config["num_concept"]
            concept_seq = batch["concept_seq"]
        else:
            num_concept = encoder_config["num_question"]
            concept_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]
        mask_seq = torch.ne(batch["mask_seq"], 0)
        ground_truth = torch.masked_select(correct_seq[:, 1:].long(), mask_seq[:, 1:])
        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()

        one_hot_vector = one_hot(concept_seq[:, 1:].long(), num_concept)
        interaction_emb = self.get_interaction_emb(batch)
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(interaction_emb)
        latent = self.attention_module(rnn_out)
        predict_score = self.sig(self.fc(self.dropout_layer(latent)))[:, :-1]
        predict_score = (predict_score * one_hot_vector).sum(-1)
        predict_score = torch.masked_select(predict_score, mask_seq[:, 1:])

        loss = 0.
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        if loss_record is not None:
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss += predict_loss

        epsilon = encoder_config["epsilon"]
        beta = encoder_config["adv loss"]

        interaction_grad = grad(predict_loss, interaction_emb, retain_graph=True)
        perturbation = torch.FloatTensor(epsilon * l2_normalize_adv(interaction_grad[0].data))
        perturbation = Variable(perturbation).to(self.params["device"])
        interaction_emb = self.get_interaction_emb(batch) + perturbation

        adv_score = self.forward_from_emb(interaction_emb)[:, :-1]
        adv_score = (adv_score * one_hot_vector).sum(-1)
        adv_score = torch.masked_select(adv_score, mask_seq[:, 1:])

        adv_loss = nn.functional.binary_cross_entropy(adv_score.double(), ground_truth.double())
        if loss_record is not None:
            loss_record.add_loss("adv loss", adv_loss.detach().cpu().item() * num_sample, num_sample)
        loss += adv_loss * beta

        return loss
