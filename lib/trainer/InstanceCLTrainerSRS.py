import torch.nn as nn

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer


class InstanceCLTrainerSRS(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(InstanceCLTrainerSRS, self).__init__(params, objects)

    def train(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        weight_cl_loss = self.params["loss_config"]["cl loss"]
        self.print_data_statics()

        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = 0.
                loss = loss + model.get_predict_loss_srs(batch, self.loss_record)
                cl_loss = model.get_cl_loss_srs(batch)
                self.loss_record.add_loss("cl loss", cl_loss.detach().cpu().item(), 1)
                loss = loss + cl_loss * weight_cl_loss
                loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break
