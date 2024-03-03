import torch
import torch.nn as nn


class DCT(nn.Module):
    model_name = "DCT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_question = encoder_config["dim_question"]
        dim_correct = encoder_config["dim_correct"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]

        self.embed_question = nn.Embedding(num_question, dim_question)
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_question + dim_correct, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_question + dim_correct, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_question + dim_correct, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.proj_latent2ability = nn.Linear(dim_latent, num_concept)
        self.proj_que2difficulty = nn.Linear(dim_question, num_concept)
        self.proj_que2discrimination = nn.Linear(dim_question, 1)

        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.proj_latent2ability.weight)
        torch.nn.init.xavier_uniform_(self.proj_que2difficulty.weight)
        torch.nn.init.xavier_uniform_(self.proj_que2discrimination.weight)

    def predict_score(self, latent, question_emb, target_question):
        user_ability = torch.sigmoid(self.proj_latent2ability(latent))
        que_difficulty = torch.sigmoid(self.proj_que2difficulty(question_emb))
        que_discrimination = torch.sigmoid(self.proj_que2discrimination(question_emb)) * 10
        # 要将target_que_concept变成可学习的一个参数
        target_que_concept = self.objects["dct"]["q_matrix"][target_question]
        y = (que_discrimination * (user_ability - que_difficulty))
        predict_score = torch.sigmoid(torch.sum(y * target_que_concept, dim=-1))

        return predict_score

    def forward(self, batch):
        dim_correct = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]["dim_correct"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)
        predict_score = self.predict_score(latent, question_emb[:, 1:], question_seq[:, 1:])

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
