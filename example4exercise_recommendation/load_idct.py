import os
import torch.nn as nn

from load_util import *


class IDCT(nn.Module):
    model_name = "IDCT"

    def __init__(self, params, objects):
        super(IDCT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["IDCT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        dropout = encoder_config["dropout"]
        q2c_table = self.objects["data"]["q2c_table"]

        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        self.question_diff_params = nn.Parameter(
            torch.zeros_like(q2c_table).float().to(self.params["device"]), requires_grad=True
        )
        self.question_disc_params = nn.Parameter(
            torch.zeros(num_question).float().to(self.params["device"]), requires_grad=True
        )
        self.encoder_layer = nn.GRU(dim_emb * 2, num_concept, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)

    def get_question_diff(self, question_seq):
        que_difficulty = torch.sigmoid(self.question_diff_params[question_seq])
        return que_difficulty

    def get_question_disc(self, question_seq):
        que_discrimination = torch.sigmoid(self.question_disc_params[question_seq])
        return que_discrimination

    def get_concept_emb(self, batch):
        q2c_table = self.objects["data"]["q2c_table"]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"]
        question_seq = batch["question_seq"]
        qc_relate = self.get_question_diff(question_seq) * q2c_mask_table[question_seq]
        sum_qc_relate = torch.sum(qc_relate, dim=-1, keepdim=True) + 1e-6
        concept_emb_relate = qc_relate.unsqueeze(-1) * self.embed_concept(q2c_table[question_seq])
        concept_emb = torch.sum(concept_emb_relate, dim=-2) / sum_qc_relate

        return concept_emb

    def cal_predict_score(self, latent, question_seq):
        max_que_disc = self.params["models_config"]["kt_model"]["encoder_layer"]["IDCT"]["max_que_disc"]
        q2c_table = self.objects["data"]["q2c_table"]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"]

        user_ability = torch.sigmoid(torch.gather(latent, 2, q2c_table[question_seq]))
        que_discrimination = self.get_question_disc(question_seq) * max_que_disc
        que_difficulty = self.get_question_diff(question_seq)
        inter_func_in = (user_ability - que_difficulty) * q2c_mask_table[question_seq]
        irt_logits = que_discrimination.unsqueeze(-1) * inter_func_in
        sum_weight_concept = torch.sum(que_difficulty * q2c_mask_table[question_seq], dim=-1, keepdim=True) + 1e-6
        y = irt_logits / sum_weight_concept
        predict_score = torch.sigmoid(torch.sum(y, dim=-1))

        return predict_score

    def get_last_user_ability(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["IDCT"]
        dim_emb = encoder_config["dim_emb"]

        correct_seq = batch["correct_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        concept_emb = self.get_concept_emb(batch)
        interaction_emb = torch.cat((concept_emb, correct_emb), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        user_ability = torch.sigmoid(latent)
        batch_size = batch["question_seq"].shape[0]
        first_index = torch.arange(batch_size).long().to(self.params["device"])
        last_ability = user_ability[first_index, batch["seq_len"] - 1]

        return last_ability

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["IDCT"]
        dim_emb = encoder_config["dim_emb"]

        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        concept_emb = self.get_concept_emb(batch)
        interaction_emb = torch.cat((concept_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        return self.cal_predict_score(latent, question_seq[:, 1:])

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)

    def get_predict_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["IDCT"]
        max_que_disc = encoder_config["max_que_disc"]
        dim_emb = encoder_config["dim_emb"]
        w_monotonic = self.params["loss_config"].get("monotonic loss", 0)
        w_mirt = self.params["loss_config"].get("mirt loss", 0)
        data_type = self.params["datasets_config"]["data_type"]

        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        concept_emb = self.get_concept_emb(batch)
        interaction_emb = torch.cat((concept_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        q2c_table = self.objects["data"]["q2c_table"]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"]
        user_ability = torch.sigmoid(torch.gather(latent, 2, q2c_table[question_seq[:, 1:]]))
        que_discrimination = self.get_question_disc(question_seq[:, 1:]) * max_que_disc
        que_difficulty = self.get_question_diff(question_seq[:, 1:])
        inter_func_in = (user_ability - que_difficulty) * q2c_mask_table[question_seq[:, 1:]]
        irt_logits = que_discrimination.unsqueeze(-1) * inter_func_in
        sum_weight_concept = torch.sum(que_difficulty * q2c_mask_table[question_seq[:, 1:]], dim=-1, keepdim=True) + 1e-6
        y = irt_logits / sum_weight_concept
        predict_score = torch.sigmoid(torch.sum(y, dim=-1))

        loss = 0.
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss = loss + predict_loss

        q2c_mask = q2c_mask_table[question_seq]
        if w_mirt != 0:
            # 单知识点习题：对于做对的题，惩罚user_ability - que_difficulty小于0的值
            #            对于做错的题，惩罚user_ability - que_difficulty大于0的值
            # 多知识点习题：对于做对的题，惩罚user_ability - que_difficulty小于0的值
            if data_type != "single_concept":
                # 对多知识点的习题损失乘一个小于1的权重
                penalty_loss_weight = self.objects["data"]["loss_weight2"]
                inter_func_in = inter_func_in * penalty_loss_weight[question_seq[:, 1:]].unsqueeze(-1)

            # 做对习题，但是在某个知识点上inter_func_in小于0
            mask4inter_func_in = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                 correct_seq[:, 1:].bool().unsqueeze(-1) & \
                                 q2c_mask[:, 1:].bool()
            target_inter_func_in1 = torch.masked_select(inter_func_in, mask4inter_func_in)

            neg_inter_func_in = target_inter_func_in1[target_inter_func_in1 < 0]
            num_sample = neg_inter_func_in.numel()

            # 对于做错的，只惩罚单知识点习题，因为使用的补偿性MIRT，对于多知识点习题不知道是哪个知识点导致做错
            is_single_concept = q2c_mask[:, 1:].sum(dim=-1) == 1
            mask4inter_func_in2 = mask_bool_seq[:, 1:] & (1 - correct_seq[:, 1:]).bool() & is_single_concept
            target_inter_func_in2 = torch.masked_select(inter_func_in[:, :, 0], mask4inter_func_in2)

            pos_inter_func_in = target_inter_func_in2[target_inter_func_in2 >= 0]
            num_sample = num_sample + pos_inter_func_in.numel()

            if num_sample > 0:
                mirt_loss = torch.cat((-neg_inter_func_in, pos_inter_func_in)).mean()
                if loss_record is not None:
                    loss_record.add_loss("mirt loss", mirt_loss.detach().cpu().item() * num_sample, num_sample)
                loss = loss + mirt_loss * w_mirt

        if w_monotonic != 0:
            # 学习约束（单调理论）：做对了题比不做题学习增长大
            user_ability_change = user_ability[:, 1:] - user_ability[:, :-1]
            if data_type != "single_concept":
                learn_loss_weight = self.objects["data"]["loss_weight1"]
                user_ability_change = user_ability_change * learn_loss_weight[question_seq[:, 1:-1]].unsqueeze(-1)

            mask4ability_change = mask_bool_seq[:, 1:-1].unsqueeze(-1) & \
                                  correct_seq[:, 1:-1].unsqueeze(-1).bool() & \
                                  q2c_mask[:, 1:-1].bool()
            target_neg_master_leval = torch.masked_select(user_ability_change, mask4ability_change)

            neg_master_leval = target_neg_master_leval[target_neg_master_leval < 0]
            num_sample = neg_master_leval.numel()
            if num_sample > 0:
                monotonic_loss = -neg_master_leval.mean()
                if loss_record is not None:
                    loss_record.add_loss("monotonic loss", monotonic_loss.detach().cpu().item() * num_sample,
                                         num_sample)
                loss = loss + monotonic_loss * w_monotonic

        return loss


def load_idct(save_model_dir, device, q_table_path, ckt_name="saved.ckt", model_name_in_ckt="best_valid"):
    global_objects = {"data": get_global_objects_data(q_table_path, device)}
    params_path = os.path.join(save_model_dir, "params.json")
    saved_params = load_json(params_path)
    saved_params["device"] = device

    ckt_path = os.path.join(save_model_dir, ckt_name)
    model = IDCT(saved_params, global_objects).to(device)
    if device == "cpu":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('cpu'))
    else:
        saved_ckt = torch.load(ckt_path)
    model.load_state_dict(saved_ckt[model_name_in_ckt])

    return model
