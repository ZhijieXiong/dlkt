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
        max_entropy_adv_aug_config = self.params["other"]["max_entropy_adv_aug"]
        use_warm_up = max_entropy_adv_aug_config["use_warm_up"]
        epoch_warm_up = max_entropy_adv_aug_config["epoch_warm_up"]
        for epoch in range(1, num_epoch + 1):
            self.do_max_entropy_aug()
            after_warm_up = epoch > epoch_warm_up

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

                if not use_warm_up or after_warm_up:
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

            # ----------------------------------------------------------------------------------------------------------
            # 每个epoch测一下模型在简单样本和困难样本上的性能
            if self.params.get("trace_epoch", False):
                model.eval()
                self.get_fine_grained_performance(model, self.objects["data_loaders"]["test_loader"], is_test=True)
                self.get_fine_grained_performance(model, self.objects["data_loaders"]["valid_loader"],
                                                  is_test=False)
                self.get_fine_grained_loss(model, self.objects["data_loaders"]["train_loader"])
            # ----------------------------------------------------------------------------------------------------------

            self.evaluate()
            if self.stop_train():
                break

        if self.params.get("trace_epoch", False):
            self.objects["logger"].info(f'AUC of valid sample')
            self.objects["logger"].info(f'AUC_seq_easy,AUC_seq_hard,AUC_question_easy,AUC_question_hard')
            for pse, psh, pqe, pqh in zip(
                    self.AUC_every_epoch_valid["seq_easy"], self.AUC_every_epoch_valid["seq_hard"],
                    self.AUC_every_epoch_valid["question_easy"], self.AUC_every_epoch_valid["question_hard"]
            ):
                self.objects["logger"].info(f"{pse},{psh},{pqe},{pqh}")

            self.objects["logger"].info(f'loss of train sample and AUC of test sample')
            self.objects["logger"].info(f'loss_seq_easy,loss_seq_hard,loss_question_easy,loss_question_hard,'
                                        f'AUC_seq_easy,AUC_seq_hard,AUC_question_easy,AUC_question_hard')
            for lse, lsh, lqe, lqh, pse, psh, pqe, pqh in zip(
                    self.loss_every_epoch["seq_easy"], self.loss_every_epoch["seq_hard"],
                    self.loss_every_epoch["question_easy"], self.loss_every_epoch["question_hard"],
                    self.AUC_every_epoch_test["seq_easy"], self.AUC_every_epoch_test["seq_hard"],
                    self.AUC_every_epoch_test["question_easy"], self.AUC_every_epoch_test["question_hard"]
            ):
                self.objects["logger"].info(f"{lse},{lsh},{lqe},{lqh},{pse},{psh},{pqe},{pqh}")
