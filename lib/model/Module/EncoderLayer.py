import torch.nn as nn


from .AKT_Block import Architecture as AKT_Architecture
from .SimpleKT_Block import Architecture as SimpleKT_Architecture


class EncoderLayer(nn.Module):
    def __init__(self, params, objects):
        super(EncoderLayer, self).__init__()
        self.params = params
        self.objects = objects

        self.encoder_type = params["models_config"]["kt_model"]["encoder_layer"]["type"]
        if self.encoder_type in ["AKT", "AKT_QUE"]:
            self.encoder = AKT_Architecture(self.params)
        elif self.encoder_type in ["SimpleKT", "SimpleKT_QUE"]:
            self.encoder = SimpleKT_Architecture(self.params)
        else:
            raise NotImplementedError()

    def forward(self, batch):
        return self.encoder(batch)

    def get_latent(self, batch):
        return self.encoder.get_latent(batch)
