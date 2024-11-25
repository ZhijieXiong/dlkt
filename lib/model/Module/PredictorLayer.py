import torch
import torch.nn as nn


class PredictorLayer(nn.Module):
    def __init__(self, params, objects):
        super(PredictorLayer, self).__init__()
        self.params = params
        self.objects = objects

        predict_layer_config = self.params["models_config"]["kt_model"]["predict_layer"]
        if predict_layer_config["type"] == "direct":
            predict_layer_config = predict_layer_config["direct"]
            dropout = predict_layer_config["dropout"]
            num_predict_layer = predict_layer_config["num_predict_layer"]
            dim_predict_in = predict_layer_config["dim_predict_in"]
            dim_predict_mid = predict_layer_config["dim_predict_mid"]
            activate_type = predict_layer_config["activate_type"]
            if activate_type == "tanh":
                act_func = nn.Tanh
            elif activate_type == "relu":
                act_func = nn.ReLU
            else:
                act_func = nn.Sigmoid
            dim_predict_out = predict_layer_config["dim_predict_out"]
            self.predict_layer = []
            if num_predict_layer == 1:
                self.predict_layer.append(nn.Dropout(dropout))
                self.predict_layer.append(nn.Linear(dim_predict_in, dim_predict_out))
                self.predict_layer.append(nn.Sigmoid())
            else:
                self.predict_layer.append(nn.Linear(dim_predict_in, dim_predict_mid))
                for _ in range(num_predict_layer - 1):
                    self.predict_layer.append(act_func())
                    self.predict_layer.append(nn.Dropout(dropout))
                    self.predict_layer.append(nn.Linear(dim_predict_mid, dim_predict_mid))
                self.predict_layer.append(nn.Dropout(dropout))
                self.predict_layer.append(nn.Linear(dim_predict_mid, dim_predict_out))
                self.predict_layer.append(nn.Sigmoid())
            self.predict_layer = nn.Sequential(*self.predict_layer)
        elif predict_layer_config["type"] == "dot":
            pass
        else:
            raise NotImplementedError()

    def forward(self, batch):
        return self.predict_layer(batch)
