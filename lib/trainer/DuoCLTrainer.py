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
        test_loader = self.objects["data_loaders"]["test_loader"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        train_statics = train_loader.dataset.get_statics_kt_dataset()
        print(f"train, seq: {train_statics[0]}, sample: {train_statics[1]}, accuracy: {train_statics[2]:<.4}")
        if train_strategy["type"] == "valid_test":
            valid_statics = self.objects["data_loaders"]["valid_loader"].dataset.get_statics_kt_dataset()
            print(f"valid, seq: {valid_statics[0]}, sample: {valid_statics[1]}, accuracy: {valid_statics[2]:<.4}")
        test_statics = test_loader.dataset.get_statics_kt_dataset()
        print(f"test, seq: {test_statics[0]}, sample: {test_statics[1]}, accuracy: {test_statics[2]:<.4}")

        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                num_seq = batch["mask_seq"].shape[0]

                predict_loss = model.get_predict_loss(batch)
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

                duo_cl_loss = model.get_duo_cl_loss(batch)
                self.loss_record.add_loss("cl loss", duo_cl_loss.detach().cpu().item() * num_seq, num_seq)

                weight_duo_cl_loss = self.params["loss_config"]["cl loss"]
                loss = predict_loss + weight_duo_cl_loss * duo_cl_loss
                loss.backward()

                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                self.objects["optimizers"]["kt_model"].step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break
