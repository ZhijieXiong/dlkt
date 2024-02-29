import torch
import torch.nn as nn

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer


class KTTrainer4LPLKTPlus(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(KTTrainer4LPLKTPlus, self).__init__(params, objects)

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
        w_que_diff_pred = self.params["loss_config"]["que diff pred loss"]
        w_que_disc_pred = self.params["loss_config"]["que disc pred loss"]

        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = 0.

                # 预测损失
                loss = loss + model.get_predict_loss(batch, self.loss_record)

                # 习题diff预测损失
                if w_que_diff_pred != 0:
                    target_que4diff = self.objects["LPKT_PLUS"]["que_has_diff_ground_truth"]
                    que_diff_pred_loss = model.get_que_diff_pred_loss(target_que4diff)
                    num_que4diff = target_que4diff.shape[0]
                    self.loss_record.add_loss("que diff pred loss",
                                              que_diff_pred_loss.detach().cpu().item() * num_que4diff, num_que4diff)
                    loss = loss + que_diff_pred_loss * w_que_diff_pred

                # 习题disc预测损失
                if w_que_disc_pred != 0:
                    target_que4disc = self.objects["LPKT_PLUS"]["que_has_disc_ground_truth"]
                    que_disc_pred_loss = model.get_que_disc_pred_loss(target_que4disc)
                    num_que4disc = target_que4disc.shape[0]
                    self.loss_record.add_loss("que disc pred loss",
                                              que_disc_pred_loss.detach().cpu().item() * num_que4disc, num_que4disc)
                    loss = loss + que_disc_pred_loss * w_que_disc_pred

                loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break
