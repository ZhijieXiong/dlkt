import os
import torch


from .util import create_optimizer
from .LossRecord import LossRecord


class KnowledgeTracingTrainer:
    def __init__(self, params):
        self.params = params

        self.optimizers = {}
        self.loss_records = {}

    def init_optimizer(self):
        opt_config = self.params["optimizer_config"]
        models = self.params["models"]

        kt_opt_config = opt_config["kt_model"]
        kt_model = models["kt_model"]
        self.optimizers["kt_model"] = create_optimizer(kt_opt_config, kt_model.parameters())

        for model_name, opt_config in opt_config["other"].items():
            self.optimizers[model_name] = create_optimizer(opt_config, models[model_name])

    def init_loss_record(self):
        self.loss_records["predict loss"] = LossRecord(["predict loss"])
        loss_config = self.params.get("loss_record_config", None)
        if loss_config is not None:
            pass