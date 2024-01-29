import torch

import torch.nn as nn


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
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKVMN"]
        use_concept = encoder_config["use_concept"]
        if use_concept:
            num_item = encoder_config["num_concept"]
            item_seq = batch["concept_seq"]
        else:
            num_item = encoder_config["num_question"]
            item_seq = batch["question_seq"]

        correct_seq = batch["correct_seq"]
        batch_size = item_seq.shape[0]
        x = item_seq + num_item * correct_seq
        k = self.k_emb_layer(item_seq)
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
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
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
