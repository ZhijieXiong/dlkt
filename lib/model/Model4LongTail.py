import torch
import torch.nn as nn


class UserBranch(nn.Module):
    def __init__(self, params, objects):
        super(UserBranch, self).__init__()
        self.params = params
        self.objects = objects

        dim_latent = params["other"]["mutual_enhance4long_tail"]["dim_latent"]
        seq_L_max = self.params["other"]["seq_L_max"]

        self.MLP = nn.Linear(dim_latent, dim_latent)
        nn.init.xavier_normal_(self.MLP.weight.data)


class ItemBranch(nn.Module):
    def __init__(self, params, objects):
        super(ItemBranch, self).__init__()
        self.params = params
        self.objects = objects

        dim_question = params["other"]["mutual_enhance4long_tail"]["dim_question"]
        dim_latent = params["other"]["mutual_enhance4long_tail"]["dim_latent"]
        question_L_max = self.params["other"]["question_L_max"]

        self.MLP = nn.Linear(dim_latent, dim_question)
        nn.init.xavier_normal_(self.MLP.weight.data)

    def forward(self, kt_model, epoch):
        pass
