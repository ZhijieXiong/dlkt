import torch
import torch.nn as nn

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer


class DuoCLTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(DuoCLTrainer, self).__init__(params, objects)

    def train(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]
        cl_type = self.params["other"]["duo_cl"]["cl_type"]

        self.print_data_statics()

        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                num_seq = batch["mask_seq"].shape[0]

                predict_loss = model.get_predict_loss(batch)
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

                duo_cl_loss = model.get_duo_cl_loss(batch, cl_type)
                self.loss_record.add_loss("cl loss", duo_cl_loss.detach().cpu().item() * num_seq, num_seq)

                weight_duo_cl_loss = self.params["loss_config"]["cl loss"]
                loss = predict_loss + weight_duo_cl_loss * duo_cl_loss
                loss.backward()

                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break
