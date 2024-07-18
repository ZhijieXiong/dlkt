import torch
import torch.nn as nn

from .BaseTrainer4AB_DA import BaseTrainer4AB_DA
from ..evaluator.util import get_seq_fine_grained_sample_mask, get_question_fine_grained_sample_mask


class AdvBiasDataAugTrainer(BaseTrainer4AB_DA):
    def __init__(self, params, objects):
        super(AdvBiasDataAugTrainer, self).__init__(params, objects)

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

        weight_adv_pred_loss = self.params["loss_config"]["ada adv predict loss"]
        ablation = self.params["other"]["adv_bias_aug"]["ablation"]
        for epoch in range(1, num_epoch + 1):
            if ablation in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                self.do_adv_aug()

            model.train()
            for _, batch in enumerate(train_loader):
                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                batch_ = {
                    "question_seqs": batch["question_seq"][:, 1:].detach().tolist(),
                    "label_seqs": batch["correct_seq"][:, 1:].detach().tolist(),
                    "mask_seqs": batch["mask_seq"][:, 1:].detach().tolist()
                }
                seq_easy_mask, seq_normal_mask, seq_hard_mask = get_seq_fine_grained_sample_mask(batch_, 10, 0.4)
                train_statics_common = self.objects["data"].get("train_data_statics_common", None)
                question_easy_mask, question_normal_mask, question_hard_mask = \
                    get_question_fine_grained_sample_mask(batch_, train_statics_common, 0.4)

                if ablation in [0, 3]:
                    bias_aligned_mask = torch.BoolTensor(question_easy_mask).to(self.params["device"])
                    bias_conflicting_mask = torch.BoolTensor(question_hard_mask).to(self.params["device"])
                elif ablation in [1, 4]:
                    bias_aligned_mask = torch.BoolTensor(seq_easy_mask).to(self.params["device"])
                    bias_conflicting_mask = torch.BoolTensor(seq_hard_mask).to(self.params["device"])
                elif ablation in [2, 5, 6, 7, 8, 9]:
                    bias_aligned_mask = torch.BoolTensor(question_easy_mask).to(self.params["device"]) | \
                                        torch.BoolTensor(seq_easy_mask).to(self.params["device"])
                    bias_conflicting_mask = torch.BoolTensor(question_hard_mask).to(self.params["device"]) | \
                                            torch.BoolTensor(seq_hard_mask).to(self.params["device"])
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                predict_loss = model.get_predict_loss(batch)
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
                predict_loss.backward()

                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

                # ADA训练
                if ablation in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    optimizer.zero_grad()
                    if ablation in [0, 1, 2, 6, 8, 9]:
                        # 使用全部样本计算对抗预测损失
                        adv_aug_predict_loss = model.get_predict_loss_from_adv_data(self.dataset_adv_generated, batch)
                    else:
                        # 只用bias_conflicting计算对抗预测损失
                        num_sample = torch.sum(bias_conflicting_mask).item()
                        adv_aug_predict_loss = model.get_predict_loss_from_adv_data(
                            self.dataset_adv_generated, batch, bias_conflicting_mask
                        )
                    self.loss_record.add_loss(
                        "ada adv predict loss", adv_aug_predict_loss.detach().cpu().item() * num_sample, num_sample
                    )
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
                self.get_fine_grained_performance(model, self.objects["data_loaders"]["valid_loader"], is_test=False)
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