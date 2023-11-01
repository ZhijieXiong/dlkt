import torch
import torch.nn as nn


class PredictorLayer(nn.Module):
    def __init__(self, params, objects):
        super(PredictorLayer, self).__init__()
        self.params = params
        self.objects = objects

