import numpy as np


class LossRecord:
    def __init__(self, loss_names):
        self.loss_all = {
            loss_name: {
                "loss_all": [],
                "num_sample_all": [],
            } for loss_name in loss_names
        }

    def clear_loss(self):
        for v in self.loss_all.values():
            v["loss_all"] = []
            v["num_sample_all"] = []

    def get_str(self):
        loss_str = ""
        for loss_name in self.loss_all:
            if len(self.loss_all[loss_name]["loss_all"]) == 0:
                loss_mean = 0.
            else:
                loss_mean = np.sum(self.loss_all[loss_name]["loss_all"]) / \
                            np.sum(self.loss_all[loss_name]["num_sample_all"])
            loss_str += f"{loss_name}: {loss_mean:<12.6}, "
        return loss_str[:-2]

    def add_loss(self, loss_name, loss, num_sample):
        self.loss_all[loss_name]["loss_all"].append(loss)
        self.loss_all[loss_name]["num_sample_all"].append(num_sample)
