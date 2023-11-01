from copy import deepcopy
import torch.nn as nn

from .Module.KTEmbedLayer import EmbedLayer
from .Module.EncoderLayer import EncoderLayer


class DKT(nn.Module):
    model_name = "DKT"

    default_config = {
        "dim_emb": 64,
        "dim_latent": 64,
        "rnn_type": "gru",
        "num_rnn_layer": 1,
        "dropout": 0.3,
        "num_predict_layer": 1,
        "dim_predict_mid": 128,
        "activate_type": "sigmoid"
    }

    def __init__(self, params, objects):
        super(DKT, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = None
        self.encoder_layer = None
        self.predict_layer = None
        self.model_config = deepcopy(DKT.default_config)

        self.init_model()

    def init_model(self):
        self.embed_layer = EmbedLayer(self.params, self.objects)
        self.encoder_layer = EncoderLayer(self.params, self.objects)


        model_config = self.params["model_config"]["kt_model"]
        embed_config = model_config["embed_layer"]
        encoder_config = model_config["encoder_layer"]
        for k in self.model_config.keys():
            if k in model_config:
                self.model_config[k] = model_config[k]

        dim_emb = self.model_config["dim_emb"]
        dim_latent = self.model_config["dim_latent"]
        rnn_type = self.model_config["rnn_type"]
        num_rnn_layer = self.model_config["num_rnn_layer"]
        dropout = self.model_config["dropout"]
        num_predict_layer = self.model_config["num_predict_layer"]
        dim_predict_mid = self.model_config["dim_predict_mid"]
        activate_type = self.model_config["activate_type"]
        dim_out = self.params["data_info"]["num_concept"] if self.params["use_concept"] else \
            self.params["data_info"]["num_question"]
        if activate_type == "tanh":
            act_func = nn.Tanh
        elif activate_type == "relu":
            act_func = nn.ReLU
        else:
            act_func = nn.Sigmoid


        predict_layer = []
        if num_predict_layer == 1:
            predict_layer.append(nn.Dropout(dropout))
            predict_layer.append(nn.Linear(dim_latent, dim_out))
            predict_layer.append(act_func())
        else:
            predict_layer.append(nn.Linear(dim_latent, dim_predict_mid))
            for _ in range(num_predict_layer - 1):
                predict_layer.append(act_func())
                predict_layer.append(nn.Dropout(dropout))
                predict_layer.append(nn.Linear(dim_predict_mid, dim_predict_mid))
            predict_layer.append(nn.Dropout(dropout))
            predict_layer.append(nn.Linear(dim_latent, dim_out))
            predict_layer.append(act_func())
        self.predict_layer = nn.ModuleList(predict_layer)
