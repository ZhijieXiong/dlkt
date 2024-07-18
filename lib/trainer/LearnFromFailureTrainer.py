import torch
import torch.nn as nn

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from ..evaluator.util import get_seq_fine_grained_performance, get_question_fine_grained_performance, \
    get_seq_fine_grained_sample_mask, get_question_fine_grained_sample_mask
from .LossRecord import LossRecord


class LearnFromFailureTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super().__init__(params, objects)
        self.loss_every_epoch4biased = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }
        self.loss_every_epoch4de_biased = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }
        self.AUC_every_epoch_valid4biased = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }
        self.AUC_every_epoch_valid4de_biased = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }
        self.AUC_every_epoch_test4biased = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }
        self.AUC_every_epoch_test4de_biased = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }

    def init_loss_record(self):
        used_losses = ["predict loss of model_biased", "predict loss of model_biased_de_biased"]
        return LossRecord(used_losses)

    def train(self):
        train_strategy = self.params["train_strategy"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        grad_clip_config_biased = self.params["grad_clip_config"]["model_biased"]
        schedulers_config_biased = self.params["schedulers_config"]["model_biased"]
        grad_clip_config_de_biased = self.params["grad_clip_config"]["model_de_biased"]
        schedulers_config_de_biased = self.params["schedulers_config"]["model_de_biased"]

        optimizer_biased = self.objects["optimizers"]["model_biased"]
        optimizer_de_biased = self.objects["optimizers"]["model_de_biased"]
        scheduler_biased = self.objects["schedulers"]["model_biased"]
        scheduler_de_biased = self.objects["schedulers"]["model_de_biased"]
        model_biased = self.objects["models"]["model_biased"]
        model_de_biased = self.objects["models"]["model_de_biased"]

        self.print_data_statics()

        for epoch in range(1, num_epoch + 1):
            model_biased.train()
            model_de_biased.train()
            for batch in train_loader:
                optimizer_biased.zero_grad()
                optimizer_de_biased.zero_grad()

                loss_per_sample_biased = model_biased.get_predict_loss_per_sample(batch).cpu().detach()
                loss_per_sample_de_biased = model_de_biased.get_predict_loss_per_sample(batch).cpu().detach()
                loss_weight = loss_per_sample_biased / (loss_per_sample_biased + loss_per_sample_de_biased + 1e-8)

                loss_biased = model_biased.get_GCE_loss(batch, q=self.params["other"]["LfF"]["q"])
                loss_de_biased = (
                        model_de_biased.get_predict_loss_per_sample(batch) * loss_weight.to(self.params["device"])
                ).mean()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                self.loss_record.add_loss(
                    "predict loss of model_biased", loss_biased.detach().cpu().item() * num_sample, num_sample
                )
                self.loss_record.add_loss(
                    "predict loss of model_biased_de_biased", loss_de_biased.detach().cpu().item() * num_sample,
                    num_sample
                )

                loss = loss_biased + loss_de_biased
                loss.backward()

                if grad_clip_config_biased["use_clip"]:
                    nn.utils.clip_grad_norm_(model_biased.parameters(), grad_clip_config_biased["grad_clipped"])
                if grad_clip_config_de_biased["use_clip"]:
                    nn.utils.clip_grad_norm_(model_de_biased.parameters(), grad_clip_config_de_biased["grad_clipped"])

                optimizer_biased.step()
                optimizer_de_biased.step()

            if schedulers_config_biased["use_scheduler"]:
                scheduler_biased.step()
            if schedulers_config_de_biased["use_scheduler"]:
                scheduler_de_biased.step()

            # ----------------------------------------------------------------------------------------------------------
            # 每个epoch测一下模型在简单样本和困难样本上的性能
            if self.params.get("trace_epoch", False):
                model_biased.eval()
                model_de_biased.eval()
                self.get_fine_grained_performance_(True, self.objects["data_loaders"]["test_loader"], is_test=True)
                self.get_fine_grained_performance_(False, self.objects["data_loaders"]["test_loader"], is_test=True)
                self.get_fine_grained_loss(True, self.objects["data_loaders"]["train_loader"])
                self.get_fine_grained_loss(False, self.objects["data_loaders"]["train_loader"])
            # ----------------------------------------------------------------------------------------------------------

            self.evaluate()

            if self.stop_train():
                break

        if self.params.get("trace_epoch", False):
            self.objects["logger"].info(f'model_biased: loss of train sample and AUC of test sample')
            self.objects["logger"].info(f'loss_seq_easy,loss_seq_hard,loss_question_easy,loss_question_hard,'
                                        f'AUC_seq_easy,AUC_seq_hard,AUC_question_easy,AUC_question_hard')
            for lse, lsh, lqe, lqh, pse, psh, pqe, pqh in zip(
                    self.loss_every_epoch4biased["seq_easy"], self.loss_every_epoch4biased["seq_hard"],
                    self.loss_every_epoch4biased["question_easy"], self.loss_every_epoch4biased["question_hard"],
                    self.AUC_every_epoch_test4biased["seq_easy"], self.AUC_every_epoch_test4biased["seq_hard"],
                    self.AUC_every_epoch_test4biased["question_easy"], self.AUC_every_epoch_test4biased["question_hard"]
            ):
                self.objects["logger"].info(f"{lse},{lsh},{lqe},{lqh},{pse},{psh},{pqe},{pqh}")

            self.objects["logger"].info(f'\n\nmodel_de_biased: loss of train sample and AUC of test sample')
            self.objects["logger"].info(f'loss_seq_easy,loss_seq_hard,loss_question_easy,loss_question_hard,'
                                        f'AUC_seq_easy,AUC_seq_hard,AUC_question_easy,AUC_question_hard')
            for lse, lsh, lqe, lqh, pse, psh, pqe, pqh in zip(
                    self.loss_every_epoch4de_biased["seq_easy"], self.loss_every_epoch4de_biased["seq_hard"],
                    self.loss_every_epoch4de_biased["question_easy"], self.loss_every_epoch4de_biased["question_hard"],
                    self.AUC_every_epoch_test4de_biased["seq_easy"], self.AUC_every_epoch_test4de_biased["seq_hard"],
                    self.AUC_every_epoch_test4de_biased["question_easy"], self.AUC_every_epoch_test4de_biased["question_hard"]
            ):
                self.objects["logger"].info(f"{lse},{lsh},{lqe},{lqh},{pse},{psh},{pqe},{pqh}")

    def get_fine_grained_performance_(self, is_model_biased, data_loader, is_test=True):
        if is_model_biased:
            model = self.objects["models"]["model_biased"]
        else:
            model = self.objects["models"]["model_de_biased"]

        has_question_seq = True
        with torch.no_grad():
            result_all_batch = []
            for batch in data_loader:
                if "question_seq" not in batch.keys():
                    has_question_seq = False
                    break
                correct_seq = batch["correct_seq"]
                question_seq = batch["question_seq"]
                predict_score = model.get_predict_score_seq_len_minus1(batch)
                result_all_batch.append({
                    "question_seqs": question_seq[:, 1:].detach().cpu().numpy().tolist(),
                    "label_seqs": correct_seq[:, 1:].detach().cpu().numpy().tolist(),
                    "predict_score_seqs": predict_score.detach().cpu().numpy().tolist(),
                    "mask_seqs": batch["mask_seq"][:, 1:].detach().cpu().numpy().tolist()
                })

        if not has_question_seq:
            return

        # 历史偏差
        seq_fine_grained_performance = get_seq_fine_grained_performance(result_all_batch, 10, 0.4)
        if is_test:
            if is_model_biased:
                self.AUC_every_epoch_test4biased["seq_easy"].append(seq_fine_grained_performance["easy"]["AUC"])
                self.AUC_every_epoch_test4biased["seq_hard"].append(seq_fine_grained_performance["hard"]["AUC"])
            else:
                self.AUC_every_epoch_test4de_biased["seq_easy"].append(seq_fine_grained_performance["easy"]["AUC"])
                self.AUC_every_epoch_test4de_biased["seq_hard"].append(seq_fine_grained_performance["hard"]["AUC"])
        else:
            if is_model_biased:
                self.AUC_every_epoch_valid4biased["seq_easy"].append(seq_fine_grained_performance["easy"]["AUC"])
                self.AUC_every_epoch_valid4biased["seq_hard"].append(seq_fine_grained_performance["hard"]["AUC"])
            else:
                self.AUC_every_epoch_valid4de_biased["seq_easy"].append(seq_fine_grained_performance["easy"]["AUC"])
                self.AUC_every_epoch_valid4de_biased["seq_hard"].append(seq_fine_grained_performance["hard"]["AUC"])

        # 习题偏差
        train_statics_common = self.objects["data"].get("train_data_statics_common", None)
        if train_statics_common is not None:
            question_fine_grained_performance = get_question_fine_grained_performance(
                result_all_batch, train_statics_common, 0.4
            )
            if is_test:
                if is_model_biased:
                    self.AUC_every_epoch_test4biased["question_easy"].append(question_fine_grained_performance["easy"]["AUC"])
                    self.AUC_every_epoch_test4biased["question_hard"].append(question_fine_grained_performance["hard"]["AUC"])
                else:
                    self.AUC_every_epoch_test4de_biased["question_easy"].append(question_fine_grained_performance["easy"]["AUC"])
                    self.AUC_every_epoch_test4de_biased["question_hard"].append(question_fine_grained_performance["hard"]["AUC"])
            else:
                if is_model_biased:
                    self.AUC_every_epoch_valid4biased["question_easy"].append(question_fine_grained_performance["easy"]["AUC"])
                    self.AUC_every_epoch_valid4biased["question_hard"].append(question_fine_grained_performance["hard"]["AUC"])
                else:
                    self.AUC_every_epoch_valid4de_biased["question_easy"].append(question_fine_grained_performance["easy"]["AUC"])
                    self.AUC_every_epoch_valid4de_biased["question_hard"].append(question_fine_grained_performance["hard"]["AUC"])

    def get_fine_grained_loss(self, is_model_biased, data_loader):
        train_statics_common = self.objects["data"].get("train_data_statics_common", None)
        if is_model_biased:
            model = self.objects["models"]["model_biased"]
        else:
            model = self.objects["models"]["model_de_biased"]

        with torch.no_grad():
            seq_easy_predict_score_all = []
            seq_easy_ground_truth_all = []
            seq_hard_predict_score_all = []
            seq_hard_ground_truth_all = []

            question_easy_predict_score_all = []
            question_easy_ground_truth_all = []
            question_hard_predict_score_all = []
            question_hard_ground_truth_all = []
            for batch in data_loader:
                predict_score = model.get_predict_score_seq_len_minus1(batch)
                ground_truth = batch["correct_seq"][:, 1:]

                batch_ = {
                    "question_seqs": batch["question_seq"][:, 1:].detach().tolist(),
                    "label_seqs": batch["correct_seq"][:, 1:].detach().tolist(),
                    "mask_seqs": batch["mask_seq"][:, 1:].detach().tolist()
                }
                seq_easy_mask, seq_normal_mask, seq_hard_mask = get_seq_fine_grained_sample_mask(batch_, 10, 0.4)
                seq_easy_mask = torch.BoolTensor(seq_easy_mask).to(self.params["device"])
                seq_hard_mask = torch.BoolTensor(seq_hard_mask).to(self.params["device"])

                seq_easy_predict_score = torch.masked_select(predict_score, seq_easy_mask).double()
                seq_easy_ground_truth = torch.masked_select(ground_truth, seq_easy_mask).double()
                seq_easy_predict_score_all.append(seq_easy_predict_score)
                seq_easy_ground_truth_all.append(seq_easy_ground_truth)

                seq_hard_predict_score = torch.masked_select(predict_score, seq_hard_mask).double()
                seq_hard_ground_truth = torch.masked_select(ground_truth, seq_hard_mask).double()
                seq_hard_predict_score_all.append(seq_hard_predict_score)
                seq_hard_ground_truth_all.append(seq_hard_ground_truth)

                if train_statics_common is not None:
                    question_easy_mask, question_normal_mask, question_hard_mask = \
                        get_question_fine_grained_sample_mask(batch_, train_statics_common, 0.4)
                    question_easy_mask = torch.BoolTensor(question_easy_mask).to(self.params["device"])
                    question_hard_mask = torch.BoolTensor(question_hard_mask).to(self.params["device"])

                    question_easy_predict_score = torch.masked_select(predict_score, question_easy_mask).double()
                    question_easy_ground_truth = torch.masked_select(ground_truth, question_easy_mask).double()
                    question_easy_predict_score_all.append(question_easy_predict_score)
                    question_easy_ground_truth_all.append(question_easy_ground_truth)

                    question_hard_predict_score = torch.masked_select(predict_score, question_hard_mask).double()
                    question_hard_ground_truth = torch.masked_select(ground_truth, question_hard_mask).double()
                    question_hard_predict_score_all.append(question_hard_predict_score)
                    question_hard_ground_truth_all.append(question_hard_ground_truth)

            seq_easy_predict_score_all = torch.cat(seq_easy_predict_score_all, dim=0)
            seq_easy_ground_truth_all = torch.cat(seq_easy_ground_truth_all, dim=0)
            seq_hard_predict_score_all = torch.cat(seq_hard_predict_score_all, dim=0)
            seq_hard_ground_truth_all = torch.cat(seq_hard_ground_truth_all, dim=0)

            seq_easy_loss = nn.functional.binary_cross_entropy(seq_easy_predict_score_all, seq_easy_ground_truth_all)
            seq_hard_loss = nn.functional.binary_cross_entropy(seq_hard_predict_score_all, seq_hard_ground_truth_all)
            if is_model_biased:
                self.loss_every_epoch4biased["seq_easy"].append(seq_easy_loss.detach().cpu().item())
                self.loss_every_epoch4biased["seq_hard"].append(seq_hard_loss.detach().cpu().item())
            else:
                self.loss_every_epoch4de_biased["seq_easy"].append(seq_easy_loss.detach().cpu().item())
                self.loss_every_epoch4de_biased["seq_hard"].append(seq_hard_loss.detach().cpu().item())

            if train_statics_common is not None:
                question_easy_predict_score_all = torch.cat(question_easy_predict_score_all, dim=0)
                question_easy_ground_truth_all = torch.cat(question_easy_ground_truth_all, dim=0)
                question_hard_predict_score_all = torch.cat(question_hard_predict_score_all, dim=0)
                question_hard_ground_truth_all = torch.cat(question_hard_ground_truth_all, dim=0)

                question_easy_loss = \
                    nn.functional.binary_cross_entropy(question_easy_predict_score_all, question_easy_ground_truth_all)
                question_hard_loss = \
                    nn.functional.binary_cross_entropy(question_hard_predict_score_all, question_hard_ground_truth_all)
                if is_model_biased:
                    self.loss_every_epoch4biased["question_easy"].append(question_easy_loss.detach().cpu().item())
                    self.loss_every_epoch4biased["question_hard"].append(question_hard_loss.detach().cpu().item())
                else:
                    self.loss_every_epoch4de_biased["question_easy"].append(question_easy_loss.detach().cpu().item())
                    self.loss_every_epoch4de_biased["question_hard"].append(question_hard_loss.detach().cpu().item())
