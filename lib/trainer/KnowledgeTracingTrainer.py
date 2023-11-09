import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error

from .util import *
from ..util.basic import *
from .LossRecord import *
from .TrainRecord import *


class KnowledgeTracingTrainer:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.objects["optimizers"] = {}
        self.objects["schedulers"] = {}

        used_losses = ["predict loss"]
        loss_config = self.params["loss_config"]
        for loss_name in loss_config:
            used_losses.append(loss_name)
        self.loss_record = LossRecord(used_losses)
        self.train_record = TrainRecord(params, objects)
        self.init_trainer()

    def init_trainer(self):
        # 初始化optimizer和scheduler
        models = self.objects["models"]
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]
        optimizers_config = self.params["optimizers_config"]
        schedulers_config = self.params["schedulers_config"]

        for model_name, optimizer_config in optimizers_config.items():
            optimizers[model_name] = create_optimizer(models[model_name].parameters(), optimizer_config)

            scheduler_config = schedulers_config[model_name]
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

        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                loss = model.get_loss(batch)
                self.loss_record.add_loss("predict loss", loss.detach().cpu().item() * num_sample, num_sample)
                loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                self.objects["optimizers"]["kt_model"].step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()

            if self.train_record.stop_training():
                if train_strategy["type"] == "no valid":
                    pass
                else:
                    best_performance_str_by_valid = self.train_record.get_evaluate_result_str("valid", "valid")
                    best_performance_str_by_test = self.train_record.get_evaluate_result_str("test", "valid")
                    print(f"best valid epoch: {self.train_record.get_best_epoch('valid'):<3} , "
                          f"best test epoch: {self.train_record.get_best_epoch('test')}\n"
                          f"valid performance by best valid epoch is {best_performance_str_by_valid}\n"
                          f"test performance by best valid epoch is {best_performance_str_by_test}")
                break

    def evaluate(self):
        train_strategy = self.params["train_strategy"]
        data_loaders = self.objects["data_loaders"]
        model = self.objects["models"]["kt_model"]
        if train_strategy["type"] == "no valid":
            # 无验证集，只有测试集
            data_loader = data_loaders["test_loader"]
            test_performance = self.evaluate_kt_dataset(model, data_loader)
            self.train_record.next_epoch(test_performance)
        else:
            # 有验证集，同时在验证集和测试集上测试
            data_loader = data_loaders["valid_loader"]
            valid_performance = self.evaluate_kt_dataset(model, data_loader)
            data_loader = data_loaders["test_loader"]
            test_performance = self.evaluate_kt_dataset(model, data_loader)
            self.train_record.next_epoch(test_performance, valid_performance)
            valid_performance_str = self.train_record.get_performance_str("valid")
            test_performance_str = self.train_record.get_performance_str("test")
            print(f"{get_now_time()} epoch {self.train_record.get_current_epoch():<3} , valid performance is "
                  f"{valid_performance_str}train loss is {self.loss_record.get_str()}, test performance is "
                  f"{test_performance_str}")

    def train_with_cl(self):
        pass

    def srs_train(self):
        pass

    @staticmethod
    def evaluate_kt_dataset(model, data_loader):
        model.eval()
        with torch.no_grad():
            predict_score_all = []
            ground_truth_all = []
            for batch in data_loader:
                correct_seq = batch["correct_seq"]
                mask_bool_seq = torch.ne(batch["mask_seq"], 0)
                predict_score = model.get_predict_score(batch).detach().cpu().numpy()
                ground_truth = torch.masked_select(correct_seq[:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)
            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
            predict_label_all = [1 if p >= 0.5 else 0 for p in predict_score_all]
            AUC = roc_auc_score(y_true=ground_truth_all, y_score=predict_score_all)
            ACC = accuracy_score(y_true=ground_truth_all, y_pred=predict_label_all)
            MAE = mean_absolute_error(y_true=ground_truth_all, y_pred=predict_score_all)
            RMSE = mean_squared_error(y_true=ground_truth_all, y_pred=predict_score_all) ** 0.5
        return {"AUC": AUC, "ACC": ACC, "MAE": MAE, "RMSE": RMSE}
