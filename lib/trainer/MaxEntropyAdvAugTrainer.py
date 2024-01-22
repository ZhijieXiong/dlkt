import torch
import torch.nn as nn

from .BaseTrainer4ME_ADA import BaseTrainer4ME_ADA


class MaxEntropyAdvAugTrainer(BaseTrainer4ME_ADA):
    def __init__(self, params, objects):
        super(MaxEntropyAdvAugTrainer, self).__init__(params, objects)

    def train(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        self.print_data_statics()

        weight_adv_pred_loss = self.params["loss_config"]["adv predict loss"]
        for epoch in range(1, num_epoch + 1):
            self.do_max_entropy_aug()

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                predict_loss = model.get_predict_loss(batch)
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
                predict_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

                optimizer.zero_grad()
                adv_aug_predict_loss = model.get_predict_loss_from_adv_data(self.dataset_adv_generated, batch)
                self.loss_record.add_loss("adv predict loss",
                                          adv_aug_predict_loss.detach().cpu().item() * num_sample, num_sample)
                adv_aug_predict_loss = weight_adv_pred_loss * adv_aug_predict_loss
                adv_aug_predict_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break
