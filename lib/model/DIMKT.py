from .Module.KTEmbedLayer import KTEmbedLayer
from .util import *


class DIMKT(nn.Module):
    model_name = "DIMKT"
    use_question = True

    def __init__(self, params, objects):
        super(DIMKT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        num_question_diff = encoder_config["num_question_diff"]
        num_concept_diff = encoder_config["num_concept_diff"]
        dropout = encoder_config["dropout"]

        self.embed_question = nn.Embedding(num_question, dim_emb)
        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        self.embed_question_diff = nn.Embedding(num_question_diff, dim_emb)
        self.embed_concept_diff = nn.Embedding(num_concept_diff, dim_emb)
        self.embed_correct = nn.Embedding(2, dim_emb)

        self.generate_x_MLP = nn.Linear(4 * dim_emb, dim_emb)
        self.SDF_MLP1 = nn.Linear(dim_emb, dim_emb)
        self.SDF_MLP2 = nn.Linear(dim_emb, dim_emb)
        self.PKA_MLP1 = nn.Linear(2 * dim_emb, dim_emb)
        self.PKA_MLP2 = nn.Linear(2 * dim_emb, dim_emb)
        self.knowledge_indicator_MLP = nn.Linear(4 * dim_emb, dim_emb)
        self.dropout_layer = nn.Dropout(dropout)

        # 解析q table
        self.question_head4zero = None
        self.embed_question4zero = None
        self.embed_question_diff4zero = None
        if self.objects["data"].get("train_data_statics", False):
            self.question_head4zero = parse_question_zero_shot(self.objects["data"]["train_data_statics"],
                                                               self.objects["data"]["question2concept"],
                                                               self.objects["data"]["concept2question"])

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

    def get_concept_diff_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            diff_fuse_table = self.objects["dimkt"]["diff_fuse_table"]
            concept_diff_emb = KTEmbedLayer.other_fused_emb(
                self.embed_concept_diff,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                diff_fuse_table,
                fusion_type="mean"
            )
        else:
            concept_diff_emb = self.embed_concept_diff(batch["concept_diff_seq"])

        return concept_diff_emb

    # ------------------------------------------------------base--------------------------------------------------------

    def forward(self, batch):
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]

        question_emb = self.embed_question(batch["question_seq"])
        question_emb = self.MLP4question(question_emb)
        concept_emb = self.get_concept_emb(batch)
        concept_emb = self.MLP4concept(concept_emb)
        question_diff_emb = self.embed_question_diff(batch["question_diff_seq"])
        concept_diff_emb = self.get_concept_diff_emb(batch)
        correct_emb = self.embed_correct(batch["correct_seq"])

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        y = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len-1):
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.cat((
                question_emb[:, t+1],
                concept_emb[:, t+1],
                question_diff_emb[:, t+1],
                concept_diff_emb[:, t+1]
            ), dim=1)
            x_next = self.generate_x_MLP(input_x_next)
            y[:, t] = torch.sigmoid(torch.sum(x_next * h, dim=-1))
            latent[:, t + 1, :] = h
            h_pre = h

        return y

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)[:, :-1]
        predict_score = torch.masked_select(predict_score_batch, mask_bool_seq[:, 1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score_batch = self.forward(batch)[:, :-1]
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
