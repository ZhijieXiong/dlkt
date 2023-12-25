import torch
import torch.nn as nn


class UserBranch(nn.Module):
    def __init__(self, params, objects):
        super(UserBranch, self).__init__()
        self.params = params
        self.objects = objects

        mutual_enhance4long_tail_config = self.params["other"]["mutual_enhance4long_tail"]
        dim_latent = mutual_enhance4long_tail_config["dim_latent"]
        seq_L_max = mutual_enhance4long_tail_config["seq_L_max"]

        self.MLP = nn.Linear(dim_latent, dim_latent)
        nn.init.xavier_normal_(self.MLP.weight.data)


class ItemBranch(nn.Module):
    def __init__(self, params, objects):
        super(ItemBranch, self).__init__()
        self.params = params
        self.objects = objects

        mutual_enhance4long_tail_config = self.params["other"]["mutual_enhance4long_tail"]
        dim_question = mutual_enhance4long_tail_config["dim_question"]
        question_L_max = mutual_enhance4long_tail_config["question_L_max"]

        self.MLP = nn.Linear(dim_question, dim_question)
        nn.init.xavier_normal_(self.MLP.weight.data)
