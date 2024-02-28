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

                # 习题diff和disc预测损失
                question_unique = torch.unique(batch["question_seq"])
                que_has_diff = self.objects["LPKT_PLUS"]["que_has_diff_ground_truth"]
                target_que4diff = question_unique[torch.isin(question_unique, que_has_diff)]
                que_diff_pred_loss = model.get_que_diff_pred_loss(target_que4diff)
                num_que4diff = target_que4diff.shape[0]
                self.loss_record.add_loss("que diff pred loss",
                                          que_diff_pred_loss.detach().cpu().item() * num_que4diff, num_que4diff)
                loss = loss + que_diff_pred_loss * w_que_diff_pred

                loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break
