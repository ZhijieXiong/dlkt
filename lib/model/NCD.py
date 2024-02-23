import torch
import torch.nn as nn


class NCD(nn.Module):
    model_name = "NCD"

    def __init__(self, params, objects):
        super(NCD, self).__init__()
        self.params = params
        self.objects = objects

        backbone_config = self.params["models_config"]["cd_model"]["backbone"]["NCD"]
        num_user = backbone_config["num_user"]
        num_question = backbone_config["num_question"]
        num_concept = backbone_config["num_concept"]

        predict_config = self.params["models_config"]["cd_model"]["predict_layer"]
        dim_predict1 = predict_config["dim_predict1"]
        dim_predict2 = predict_config["dim_predict2"]
        dropout = predict_config["dropout"]

        # network structure
        self.embed_user = nn.Embedding(num_user, num_concept)
        self.embed_question_diff = nn.Embedding(num_question, num_concept)
        self.embed_question_disc = nn.Embedding(num_question, 1)
        self.predict_layer1 = nn.Linear(num_concept, dim_predict1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.predict_layer2 = nn.Linear(dim_predict1, dim_predict2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.predict_layer3 = nn.Linear(dim_predict2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        user_id = batch["user_id"]
        question_id = batch["question_id"]
        Q_table = self.objects["data"]["Q_table_tensor"]

        user_emb = torch.sigmoid(self.embed_user(user_id))
        question_diff = torch.sigmoid(self.embed_question_diff(question_id))
        question_disc = torch.sigmoid(self.embed_question_disc(question_id)) * 10
        input_x = question_disc * (user_emb - question_diff) * Q_table[question_id]
        input_x = self.drop_1(torch.sigmoid(self.predict_layer1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.predict_layer2(input_x)))
        output = torch.sigmoid(self.predict_layer3(input_x)).squeeze(dim=-1)

        return output

    def get_predict_loss(self, batch):
        predict_score = self.forward(batch)
        ground_truth = batch["correct"]
        loss = torch.nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        return loss

    def get_predict_score(self, batch):
        return self.forward(batch)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.predict_layer1.apply(clipper)
        self.predict_layer2.apply(clipper)
        self.predict_layer3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.embed_user(stu_id))
        return stat_emb.data

    def get_question_params(self, question_id):
        k_difficulty = torch.sigmoid(self.embed_question_diff(question_id))
        e_discrimination = torch.sigmoid(self.embed_question_disc(question_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
