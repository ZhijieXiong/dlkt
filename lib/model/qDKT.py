from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer
from .util import *


class qDKT(nn.Module):
    model_name = "qDKT"
    use_question = True

    def __init__(self, params, objects):
        super(qDKT, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_concept = encoder_config["dim_concept"]
        dim_question = encoder_config["dim_question"]
        dim_correct = encoder_config["dim_correct"]
        dim_emb = dim_concept + dim_question + dim_correct
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)

        self.predict_layer = PredictorLayer(self.params, self.objects)

    def get_qc_emb4single_concept(self, batch):
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]

        return self.embed_layer.get_emb_concatenated(("concept", "question"),
                                                     (concept_seq, question_seq))

    def get_qc_emb4only_question(self, batch):
        question_seq = batch["question_seq"]

        return self.embed_layer.get_emb_question_with_concept_fused(question_seq, fusion_type="mean")

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]
        data_type = self.params["datasets_config"]["data_type"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question(batch)
        else:
            qc_emb = self.get_qc_emb4single_concept(batch)
        interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        predict_layer_input = torch.cat((latent, qc_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch, mask_bool_seq[:, 1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch, mask_bool_seq[:, 1:])
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        # 判断是否有样本损失权重设置
        sample_weight = None
        if self.params.get("sample_reweight", False) and self.params["sample_reweight"].get("use_sample_reweight", False):
            sample_weight = torch.masked_select(batch["weight_seq"][:, 1:], mask_bool_seq[:, 1:])

        # 计算损失
        predict_loss = nn.functional.binary_cross_entropy(
            predict_score.double(), ground_truth.double(), weight=sample_weight
        )

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": predict_loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_score4target_question(self, latent, question_seq, concept_seq=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_concept = encoder_config["dim_concept"]
        dim_question = encoder_config["dim_question"]
        dim_latent = encoder_config["dim_latent"]
        data_type = self.params["datasets_config"]["data_type"]
        num_latent = len(latent)
        num_question = len(question_seq)

        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question({"question_seq": question_seq})
        else:
            qc_emb = self.get_qc_emb4single_concept({
                "question_seq": question_seq,
                "concept_seq": concept_seq
            })
        predict_layer_input = torch.cat((
            latent.repeat(1, num_question).view(num_latent, num_question, dim_latent),
            qc_emb.view(1, num_question, dim_concept + dim_question).repeat(num_latent, 1, 1)
        ), dim=-1)

        return self.predict_layer(predict_layer_input).squeeze(dim=-1)

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        predict_score = self.forward(batch)
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"][:, 1:], penultimate=False)

        return predict_score[mask4last.bool()]
