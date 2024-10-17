import torch
import os
import torch.nn as nn

from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error
from copy import deepcopy

from .util import *
from ..util.basic import *
from .LossRecord import *
from .TrainRecord import *
from ..evaluator.util import get_seq_fine_grained_performance, get_question_fine_grained_performance, \
    get_seq_fine_grained_sample_mask, get_question_fine_grained_sample_mask


class KnowledgeTracingTrainer:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.best_model = None
        self.objects["optimizers"] = {}
        self.objects["schedulers"] = {}

        self.loss_every_epoch = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }
        self.AUC_every_epoch_valid = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }
        self.AUC_every_epoch_test = {
            "seq_easy": [],
            "seq_hard": [],
            "question_easy": [],
            "question_hard": []
        }

        self.loss_record = self.init_loss_record()
        self.train_record = TrainRecord(params, objects)
        self.init_trainer()

    def init_loss_record(self):
        use_mix_up = self.params.get("use_mix_up", False)
        used_losses = ["predict loss"]
        if use_mix_up:
            used_losses.append("mix up predict loss")
        loss_config = self.params["loss_config"]
        for loss_name in loss_config:
            used_losses.append(loss_name)
        return LossRecord(used_losses)

    def init_trainer(self):
        # 初始化optimizer和scheduler
        models = self.objects["models"]
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]
        optimizers_config = self.params["optimizers_config"]
        schedulers_config = self.params["schedulers_config"]

        for model_name, optimizer_config in optimizers_config.items():
            if models[model_name] is not None:
                scheduler_config = schedulers_config[model_name]
                optimizers[model_name] = create_optimizer(models[model_name].parameters(), optimizer_config)

                if scheduler_config["use_scheduler"]:
                    schedulers[model_name] = create_scheduler(optimizers[model_name], scheduler_config)
                else:
                    schedulers[model_name] = None

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

        for epoch in range(1, num_epoch + 1):
            model.train()

            # 用于追踪每个epoch的细粒度predict loss指标
            seq_easy_predict_score_all = []
            seq_easy_ground_truth_all = []
            seq_hard_predict_score_all = []
            seq_hard_ground_truth_all = []

            question_easy_predict_score_all = []
            question_easy_ground_truth_all = []
            question_hard_predict_score_all = []
            question_hard_ground_truth_all = []

            for batch in train_loader:
                if model.model_name == "DTransformer":
                    batch["concept_seq"] = batch["concept_seq"].masked_fill(torch.eq(batch["mask_seq"], 0), -1)
                    batch["question_seq"] = batch["question_seq"].masked_fill(torch.eq(batch["mask_seq"], 0), -1)
                    batch["correct_seq"] = batch["correct_seq"].masked_fill(torch.eq(batch["mask_seq"], 0), -1)
                optimizer.zero_grad()
                loss_result = model.get_predict_loss(batch)
                for loss_name, loss_info in loss_result["losses_value"].items():
                    if loss_name != "total loss":
                        self.loss_record.add_loss(loss_name, loss_info["value"], loss_info["num_sample"])
                loss_result["total_loss"].backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

                if self.params.get("trace_epoch", False):
                    # 追踪每个epoch的细粒度损失
                    model.eval()
                    fine_grained_loss_result = self.get_fine_grained_loss(batch, loss_result["predict_score_batch"])
                    seq_easy_predict_score_all.append(fine_grained_loss_result["seq_easy_predict_score"])
                    seq_easy_ground_truth_all.append(fine_grained_loss_result["seq_easy_ground_truth"])
                    seq_hard_predict_score_all.append(fine_grained_loss_result["seq_hard_predict_score"])
                    seq_hard_ground_truth_all.append(fine_grained_loss_result["seq_hard_ground_truth"])

                    if "question_easy_predict_score" in fine_grained_loss_result:
                        question_easy_predict_score_all.append(fine_grained_loss_result["question_easy_predict_score"])
                        question_easy_ground_truth_all.append(fine_grained_loss_result["question_easy_ground_truth"])
                        question_hard_predict_score_all.append(fine_grained_loss_result["question_hard_predict_score"])
                        question_hard_ground_truth_all.append(fine_grained_loss_result["question_hard_ground_truth"])
                    model.train()

            if schedulers_config["use_scheduler"]:
                scheduler.step()

            # ----------------------------------------------------------------------------------------------------------
            # 获取在细粒度样本上的损失
            if self.params.get("trace_epoch", False):
                model.eval()
                # 打印细粒度的损失
                seq_easy_predict_score_all = torch.cat(seq_easy_predict_score_all, dim=0)
                seq_easy_ground_truth_all = torch.cat(seq_easy_ground_truth_all, dim=0)
                seq_hard_predict_score_all = torch.cat(seq_hard_predict_score_all, dim=0)
                seq_hard_ground_truth_all = torch.cat(seq_hard_ground_truth_all, dim=0)

                seq_easy_loss = nn.functional.binary_cross_entropy(
                    seq_easy_predict_score_all, seq_easy_ground_truth_all
                )
                self.loss_every_epoch["seq_easy"].append(seq_easy_loss.detach().cpu().item())

                seq_hard_loss = nn.functional.binary_cross_entropy(
                    seq_hard_predict_score_all, seq_hard_ground_truth_all
                )
                self.loss_every_epoch["seq_hard"].append(seq_hard_loss.detach().cpu().item())

                if len(question_easy_predict_score_all) > 0:
                    question_easy_predict_score_all = torch.cat(question_easy_predict_score_all, dim=0)
                    question_easy_ground_truth_all = torch.cat(question_easy_ground_truth_all, dim=0)
                    question_hard_predict_score_all = torch.cat(question_hard_predict_score_all, dim=0)
                    question_hard_ground_truth_all = torch.cat(question_hard_ground_truth_all, dim=0)

                    question_easy_loss = nn.functional.binary_cross_entropy(
                        question_easy_predict_score_all, question_easy_ground_truth_all
                    )
                    self.loss_every_epoch["question_easy"].append(question_easy_loss.detach().cpu().item())

                    question_hard_loss = nn.functional.binary_cross_entropy(
                        question_hard_predict_score_all, question_hard_ground_truth_all
                    )
                    self.loss_every_epoch["question_hard"].append(question_hard_loss.detach().cpu().item())
            # ----------------------------------------------------------------------------------------------------------

            self.evaluate()
            if self.stop_train():
                break

        if self.params.get("trace_epoch", False):
            self.objects["logger"].info(f'{"-"*35}AUC of valid sample in fine-grained sample{"-"*35}')
            # 历史偏差
            self.objects["logger"].info(f'AUC_seq_easy,AUC_seq_hard')
            for pse, psh in zip(
                    self.AUC_every_epoch_valid["seq_easy"], self.AUC_every_epoch_valid["seq_hard"],
            ):
                self.objects["logger"].info(f"{pse},{psh}")
            # 习题偏差，可能没有，需要有static common数据
            self.objects["logger"].info(f'AUC_question_easy,AUC_question_hard')
            for pqe, pqh in zip(
                    self.AUC_every_epoch_valid["question_easy"], self.AUC_every_epoch_valid["question_hard"]
            ):
                self.objects["logger"].info(f"{pqe},{pqh}")
            self.objects["logger"].info(f'{"-" * 100}')

            self.objects["logger"].info(f'{"-"*25}loss (train) and AUC (test) in fine-grained sample{"-"*25}')
            # 历史偏差
            self.objects["logger"].info(f'loss_seq_easy,loss_seq_hard,AUC_seq_easy,AUC_seq_hard')
            for lse, lsh, pse, psh in zip(
                    self.loss_every_epoch["seq_easy"], self.loss_every_epoch["seq_hard"],
                    self.AUC_every_epoch_test["seq_easy"], self.AUC_every_epoch_test["seq_hard"],
            ):
                self.objects["logger"].info(f"{lse},{lsh},{pse},{psh}")
            # 习题偏差，可能没有，需要有static common数据
            self.objects["logger"].info(f'loss_question_easy,loss_question_hard,AUC_question_easy,AUC_question_hard')
            for lqe, lqh, pqe, pqh in zip(
                    self.loss_every_epoch["question_easy"], self.loss_every_epoch["question_hard"],
                    self.AUC_every_epoch_test["question_easy"], self.AUC_every_epoch_test["question_hard"]
            ):
                self.objects["logger"].info(f"{lqe},{lqh},{pqe},{pqh}")
            self.objects["logger"].info(f'{"-" * 100}')

    def stop_train(self):
        train_strategy = self.params["train_strategy"]
        stop_flag = self.train_record.stop_training()
        if stop_flag:
            if train_strategy["type"] == "no_valid":
                pass
            else:
                best_train_performance_by_valid = self.train_record.get_evaluate_result_str("train", "valid")
                best_valid_performance_by_valid = self.train_record.get_evaluate_result_str("valid", "valid")
                best_test_performance_by_valid = self.train_record.get_evaluate_result_str("test", "valid")

                self.objects["logger"].info(
                    f"best valid epoch: {self.train_record.get_best_epoch('valid'):<3} , "
                    f"best test epoch: {self.train_record.get_best_epoch('test')}\n"
                    f"train performance by best valid epoch is {best_train_performance_by_valid}\n"
                    f"valid performance by best valid epoch is {best_valid_performance_by_valid}\n"
                    f"test performance by best valid epoch is {best_test_performance_by_valid}\n"
                    f"{'-'*100}\n"
                    f"train performance by best train epoch is "
                    f"{self.train_record.get_evaluate_result_str('train', 'train')}\n"
                    f"test performance by best test epoch is "
                    f"{self.train_record.get_evaluate_result_str('test', 'test')}\n"
                )

                valid_loader = self.objects["data_loaders"]["valid_loader"]
                test_loader = self.objects["data_loaders"]["test_loader"]
                model = self.objects["models"]["kt_model"]
                if model.use_question:
                    len1 = len("fine-grained metric of valid data")
                    self.objects["logger"].info("-" * ((100 - len1) // 2) + "fine-grained metric of valid data" +
                                                "-" * ((100 - len1) // 2))
                    self.evaluate_fine_grained(valid_loader, "valid")
                    self.objects["logger"].info("-" * 100)

                    len2 = len("fine-grained metric of test data")
                    self.objects["logger"].info("-" * ((100 - len2) // 2) + "fine-grained metric of test data" +
                                                "-" * ((100 - len2) // 2))
                    self.evaluate_fine_grained(test_loader, "test")
                    self.objects["logger"].info("-" * 100)

        return stop_flag

    def print_performance(self, prefix, performance_dict):
        self.objects["logger"].info(
            f"{prefix}"
            f"AUC: {performance_dict['AUC']:<9.6}, "
            f"ACC: {performance_dict['ACC']:<9.6}, "
            f"RMSE: {performance_dict['RMSE']:<9.6}, "
            f"MAE: {performance_dict['MAE']:<9.6}"
        )

    def evaluate_fine_grained(self, data_loader, valid_or_test):
        model = self.best_model.to(self.params["device"])
        model.eval()
        with torch.no_grad():
            result_all_batch = []
            for batch in data_loader:
                correct_seq = batch["correct_seq"]
                question_seq = batch["question_seq"]
                predict_score_batch = model.get_predict_score(batch)["predict_score_batch"]
                result_all_batch.append({
                    "question_seqs": question_seq[:, 1:].detach().cpu().numpy().tolist(),
                    "label_seqs": correct_seq[:, 1:].detach().cpu().numpy().tolist(),
                    "predict_score_seqs": predict_score_batch.detach().cpu().numpy().tolist(),
                    "mask_seqs": batch["mask_seq"][:, 1:].detach().cpu().numpy().tolist()
                })

        window_lens = [10, 20]
        acc_ths = [0.4, 0.4]
        for window_len, acc_th in zip(window_lens, acc_ths):
            seq_fine_grained_performance = get_seq_fine_grained_performance(result_all_batch, window_len, acc_th)
            self.print_performance(
                f"({window_len}, {acc_th}) {valid_or_test} seq easy point, "
                f"performance is ", seq_fine_grained_performance['easy']
            )
            self.print_performance(
                f"({window_len}, {acc_th}) {valid_or_test} seq normal point, "
                f"performance is ", seq_fine_grained_performance['normal']
            )
            self.print_performance(
                f"({window_len}, {acc_th}) {valid_or_test} seq hard point, "
                f"performance is ", seq_fine_grained_performance['hard']
            )
            self.print_performance(
                f"({window_len}, {acc_th}) {valid_or_test} seq normal&hard point, "
                f"performance is ", seq_fine_grained_performance['normal&hard']
            )

        # 习题偏差
        train_statics_common = self.objects["data"].get("train_data_statics_common", None)
        if train_statics_common is not None:
            acc_ths = [0.4, 0.3, 0.2]
            for acc_th in acc_ths:
                question_fine_grained_performance = get_question_fine_grained_performance(
                    result_all_batch, train_statics_common, acc_th
                )
                self.print_performance(
                    f"({acc_th}, ) {valid_or_test} question easy point, "
                    f"performance is ", question_fine_grained_performance['easy']
                )
                self.print_performance(
                    f"({acc_th}, ) {valid_or_test} question normal point, "
                    f"performance is ", question_fine_grained_performance['normal']
                )
                self.print_performance(
                    f"({acc_th}, ) {valid_or_test} question hard point, "
                    f"performance is ", question_fine_grained_performance['hard']
                )
                self.print_performance(
                    f"({acc_th}, ) {valid_or_test} question normal&hard point, "
                    f"performance is ", question_fine_grained_performance['normal&hard']
                )

    def evaluate(self):
        train_strategy = self.params["train_strategy"]
        save_model = self.params["save_model"]
        data_loaders = self.objects["data_loaders"]
        train_loader = data_loaders["train_loader"]
        model = self.objects["models"]["kt_model"]
        train_performance = self.evaluate_kt_dataset(model, train_loader)
        if train_strategy["type"] == "no_valid":
            # 无验证集，只有测试集
            data_loader = data_loaders["test_loader"]
            test_performance = self.evaluate_kt_dataset(model, data_loader)
            self.train_record.next_epoch(train_performance, test_performance)
        else:
            # 有验证集，同时在验证集和测试集上测试
            data_loader = data_loaders["valid_loader"]
            valid_performance = self.evaluate_kt_dataset(model, data_loader)
            if self.params.get("trace_epoch", False):
                # 存储历史细粒度性能信息
                self.AUC_every_epoch_valid["seq_easy"].append(
                    valid_performance["fine_grained_performance"]["seq_easy"]["AUC"]
                )
                self.AUC_every_epoch_valid["seq_hard"].append(
                    valid_performance["fine_grained_performance"]["seq_hard"]["AUC"]
                )
                if "question_easy" in valid_performance["fine_grained_performance"]:
                    # 存储习题细粒度性能信息
                    self.AUC_every_epoch_valid["question_easy"].append(
                        valid_performance["fine_grained_performance"]["question_easy"]["AUC"]
                    )
                    self.AUC_every_epoch_valid["question_hard"].append(
                        valid_performance["fine_grained_performance"]["question_hard"]["AUC"]
                    )

            data_loader = data_loaders["test_loader"]
            test_performance = self.evaluate_kt_dataset(model, data_loader)
            if self.params.get("trace_epoch", False):
                # 存储细粒度性能信息
                self.AUC_every_epoch_test["seq_easy"].append(
                    test_performance["fine_grained_performance"]["seq_easy"]["AUC"]
                )
                self.AUC_every_epoch_test["seq_hard"].append(
                    test_performance["fine_grained_performance"]["seq_hard"]["AUC"]
                )
                if "question_easy" in test_performance["fine_grained_performance"]:
                    # 存储习题细粒度性能信息
                    self.AUC_every_epoch_valid["question_easy"].append(
                        test_performance["fine_grained_performance"]["question_easy"]["AUC"]
                    )
                    self.AUC_every_epoch_valid["question_hard"].append(
                        test_performance["fine_grained_performance"]["question_hard"]["AUC"]
                    )

            self.train_record.next_epoch(train_performance, test_performance, valid_performance)
            valid_performance_str = self.train_record.get_performance_str("valid")
            test_performance_str = self.train_record.get_performance_str("test")
            best_epoch = self.train_record.get_best_epoch("valid")
            self.objects["logger"].info(
                f"{get_now_time()} epoch {self.train_record.get_current_epoch():<3} , valid performance is "
                f"{valid_performance_str}train loss is {self.loss_record.get_str()}, test performance is "
                f"{test_performance_str}current best epoch is {best_epoch}")
            self.loss_record.clear_loss()
            current_epoch = self.train_record.get_current_epoch()
            if best_epoch == current_epoch:
                # 节省显存
                self.best_model = deepcopy(model).to("cpu")
                if save_model:
                    save_model_dir = self.params["save_model_dir"]
                    model_weight_path = os.path.join(save_model_dir, "saved.ckt")
                    torch.save({"best_valid": model.state_dict()}, model_weight_path)

    def print_data_statics(self):
        train_strategy = self.params["train_strategy"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        test_loader = self.objects["data_loaders"]["test_loader"]

        self.objects["logger"].info("")
        train_statics = train_loader.dataset.get_statics_kt_dataset()
        self.objects["logger"].info(f"train, seq: {train_statics[0]}, sample: {train_statics[1]}, accuracy: {train_statics[2]:<.4}")
        if train_strategy["type"] == "valid_test":
            valid_statics = self.objects["data_loaders"]["valid_loader"].dataset.get_statics_kt_dataset()
            self.objects["logger"].info(f"valid, seq: {valid_statics[0]}, sample: {valid_statics[1]}, accuracy: {valid_statics[2]:<.4}")
        test_statics = test_loader.dataset.get_statics_kt_dataset()
        self.objects["logger"].info(f"test, seq: {test_statics[0]}, sample: {test_statics[1]}, accuracy: {test_statics[2]:<.4}")

    def evaluate_kt_dataset(self, model, data_loader):
        model.eval()
        performance_result = {}
        with torch.no_grad():
            result_all_batch = []
            predict_score_all = []
            ground_truth_all = []
            for batch in data_loader:
                correct_seq = batch["correct_seq"]
                mask_bool_seq = torch.ne(batch["mask_seq"], 0)
                score_result = model.get_predict_score(batch)
                predict_score = score_result["predict_score"].detach().cpu().numpy()
                ground_truth = torch.masked_select(correct_seq[:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)

                # 获取细粒度性能所需的数据
                if "question_seq" in batch.keys():
                    correct_seq = batch["correct_seq"]
                    question_seq = batch["question_seq"]
                    result_all_batch.append({
                        "question_seqs": question_seq[:, 1:].detach().cpu().numpy().tolist(),
                        "label_seqs": correct_seq[:, 1:].detach().cpu().numpy().tolist(),
                        "predict_score_seqs": score_result["predict_score_batch"].detach().cpu().numpy().tolist(),
                        "mask_seqs": batch["mask_seq"][:, 1:].detach().cpu().numpy().tolist()
                    })

            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
            predict_label_all = [1 if p >= 0.5 else 0 for p in predict_score_all]
            AUC = roc_auc_score(y_true=ground_truth_all, y_score=predict_score_all)
            ACC = accuracy_score(y_true=ground_truth_all, y_pred=predict_label_all)
            MAE = mean_absolute_error(y_true=ground_truth_all, y_pred=predict_score_all)
            RMSE = mean_squared_error(y_true=ground_truth_all, y_pred=predict_score_all) ** 0.5

            performance_result["AUC"] = AUC
            performance_result["ACC"] = ACC
            performance_result["MAE"] = MAE
            performance_result["RMSE"] = RMSE

            if self.params.get("trace_epoch", False):
                fine_grained_performance = self.get_fine_grained_performance(result_all_batch)
                performance_result["fine_grained_performance"] = fine_grained_performance

        return performance_result

    def get_fine_grained_performance(self, result_all_batch):
        fine_grained_performance = {}

        # 历史偏差
        seq_fine_grained_performance = get_seq_fine_grained_performance(result_all_batch, 10, 0.4)
        fine_grained_performance["seq_easy"] = {
            "AUC": seq_fine_grained_performance["easy"]["AUC"],
            "ACC": seq_fine_grained_performance["easy"]["ACC"],
            "MAE": seq_fine_grained_performance["easy"]["MAE"],
            "RMSE": seq_fine_grained_performance["easy"]["RMSE"]
        }
        fine_grained_performance["seq_hard"] = {
            "AUC": seq_fine_grained_performance["hard"]["AUC"],
            "ACC": seq_fine_grained_performance["hard"]["ACC"],
            "MAE": seq_fine_grained_performance["hard"]["MAE"],
            "RMSE": seq_fine_grained_performance["hard"]["RMSE"]
        }

        # 习题偏差
        train_statics_common = self.objects["data"].get("train_data_statics_common", None)
        if train_statics_common is not None:
            question_fine_grained_performance = get_question_fine_grained_performance(
                result_all_batch, train_statics_common, 0.4
            )
            fine_grained_performance["question_easy"] = {
                "AUC": question_fine_grained_performance["easy"]["AUC"],
                "ACC": question_fine_grained_performance["easy"]["ACC"],
                "MAE": question_fine_grained_performance["easy"]["MAE"],
                "RMSE": question_fine_grained_performance["easy"]["RMSE"]
            }
            fine_grained_performance["question_hard"] = {
                "AUC": question_fine_grained_performance["hard"]["AUC"],
                "ACC": question_fine_grained_performance["hard"]["ACC"],
                "MAE": question_fine_grained_performance["hard"]["MAE"],
                "RMSE": question_fine_grained_performance["hard"]["RMSE"]
            }

        return fine_grained_performance

    def get_fine_grained_loss(self, batch, predict_score_batch):
        train_statics_common = self.objects["data"].get("train_data_statics_common", None)

        with torch.no_grad():
            fine_grained_data = {}
            ground_truth = batch["correct_seq"][:, 1:]
            batch_ = {
                "question_seqs": batch["question_seq"][:, 1:].detach().tolist(),
                "label_seqs": batch["correct_seq"][:, 1:].detach().tolist(),
                "mask_seqs": batch["mask_seq"][:, 1:].detach().tolist()
            }
            seq_easy_mask, seq_normal_mask, seq_hard_mask = get_seq_fine_grained_sample_mask(batch_, 10, 0.4)
            seq_easy_mask = torch.BoolTensor(seq_easy_mask).to(self.params["device"])
            seq_hard_mask = torch.BoolTensor(seq_hard_mask).to(self.params["device"])

            seq_easy_predict_score = torch.masked_select(predict_score_batch, seq_easy_mask).double()
            seq_easy_ground_truth = torch.masked_select(ground_truth, seq_easy_mask).double()
            fine_grained_data["seq_easy_predict_score"] = seq_easy_predict_score
            fine_grained_data["seq_easy_ground_truth"] = seq_easy_ground_truth

            seq_hard_predict_score = torch.masked_select(predict_score_batch, seq_hard_mask).double()
            seq_hard_ground_truth = torch.masked_select(ground_truth, seq_hard_mask).double()
            fine_grained_data["seq_hard_predict_score"] = seq_hard_predict_score
            fine_grained_data["seq_hard_ground_truth"] = seq_hard_ground_truth

            if train_statics_common is not None:
                question_easy_mask, question_normal_mask, question_hard_mask = \
                    get_question_fine_grained_sample_mask(batch_, train_statics_common, 0.4)
                question_easy_mask = torch.BoolTensor(question_easy_mask).to(self.params["device"])
                question_hard_mask = torch.BoolTensor(question_hard_mask).to(self.params["device"])

                question_easy_predict_score = torch.masked_select(predict_score_batch, question_easy_mask).double()
                question_easy_ground_truth = torch.masked_select(ground_truth, question_easy_mask).double()
                fine_grained_data["question_easy_predict_score"] = question_easy_predict_score
                fine_grained_data["question_easy_ground_truth"] = question_easy_ground_truth

                question_hard_predict_score = torch.masked_select(predict_score_batch, question_hard_mask).double()
                question_hard_ground_truth = torch.masked_select(ground_truth, question_hard_mask).double()
                fine_grained_data["question_hard_predict_score"] = question_hard_predict_score
                fine_grained_data["question_hard_ground_truth"] = question_hard_ground_truth

            return fine_grained_data
