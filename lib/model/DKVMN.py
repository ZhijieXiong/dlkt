import torch
import torch.nn as nn

from .Module.KTEmbedLayer import KTEmbedLayer


class DKVMN(nn.Module):
    model_name = "DKVMN"

    def __init__(self, params, objects):
        super(DKVMN, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["DKVMN"]
        use_concept = encoder_config["use_concept"]
        if use_concept:
            num_item = encoder_config["num_concept"]
        else:
            num_item = encoder_config["num_question"]
        dim_key = encoder_config["dim_key"]
        dim_value = encoder_config["dim_value"]
        dropout = encoder_config["dropout"]

        self.k_emb_layer = nn.Embedding(num_item, dim_key)
        self.Mk = nn.Parameter(torch.Tensor(dim_value, dim_key))
        self.Mv0 = nn.Parameter(torch.Tensor(dim_value, dim_key))

        nn.init.kaiming_normal_(self.Mk)
        nn.init.kaiming_normal_(self.Mv0)

        self.v_emb_layer = nn.Embedding(num_item * 2, dim_key)

        self.f_layer = nn.Linear(dim_key * 2, dim_key)
        self.dropout_layer = nn.Dropout(dropout)
        self.p_layer = nn.Linear(dim_key, 1)

        self.e_layer = nn.Linear(dim_key, dim_key)
        self.a_layer = nn.Linear(dim_key, dim_key)

    def forward(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKVMN"]
        use_concept = encoder_config["use_concept"]
        correct_seq = batch["correct_seq"]
        batch_size = correct_seq.shape[0]

        if use_concept:
            num_concept = encoder_config["num_concept"]
            if data_type == "single_concept":
                concept_seq = batch["concept_seq"]
                x = concept_seq + num_concept * correct_seq
                k = self.k_emb_layer(concept_seq)
                v = self.v_emb_layer(x)
            else:
                question_seq = batch["question_seq"]
                k = KTEmbedLayer.concept_fused_emb(
                    self.k_emb_layer,
                    self.objects["data"]["q2c_table"],
                    self.objects["data"]["q2c_mask_table"],
                    question_seq,
                    fusion_type="mean"
                )
                v = KTEmbedLayer.interaction_fused_emb(
                    self.v_emb_layer,
                    self.objects["data"]["q2c_table"],
                    self.objects["data"]["q2c_mask_table"],
                    question_seq,
                    correct_seq,
                    num_concept,
                    fusion_type="mean"
                )
        else:
            num_question = encoder_config["num_question"]
            question_seq = batch["question_seq"]
            x = question_seq + num_question * correct_seq
            k = self.k_emb_layer(question_seq)
            v = self.v_emb_layer(x)

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat([(w.unsqueeze(-1) * Mv[:, :-1]).sum(-2), k], dim=-1)
            )
        )

        predict_score = torch.sigmoid(self.p_layer(self.dropout_layer(f))).squeeze(-1)

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
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)[:, 1:]
