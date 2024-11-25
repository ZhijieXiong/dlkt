class TrainRecord:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.record = {
            "current_epoch": 0,
            "best_train_main_metric": 0,
            "best_valid_main_metric": 0,
            "best_test_main_metric": 0,
            "best_epoch_by_train": 1,
            "best_epoch_by_valid": 1,
            "best_epoch_by_test": 1,
            "performance_train": [],
            "performance_valid": [],
            "performance_test": []
        }

    def next_epoch(self, train_performance, valid_performance, test_performance=None):
        self.record["current_epoch"] += 1
        train_strategy = self.params["train_strategy"]
        self.update_best_metric(train_performance, update_type="train")
        self.update_best_metric(valid_performance, update_type="valid")
        if train_strategy["type"] == "valid_test":
            self.update_best_metric(test_performance, update_type="test")

    def get_current_epoch(self):
        return self.record["current_epoch"]

    def get_best_epoch(self, performance_by):
        if performance_by == "train":
            performance_index = self.record["best_epoch_by_train"]
        elif performance_by == "valid":
            performance_index = self.record["best_epoch_by_valid"]
        else:
            performance_index = self.record["best_epoch_by_test"]
        return performance_index

    def update_best_metric(self, performance, update_type):
        use_multi_metrics = self.params["train_strategy"]["use_multi_metrics"]
        main_metric_key = self.params["train_strategy"]["main_metric"]
        multi_metrics = self.params["train_strategy"]["multi_metrics"]
        main_metric = self.cal_main_metric(performance, multi_metrics) if use_multi_metrics else (
            performance)[main_metric_key]
        performance["main_metric"] = main_metric

        if update_type == "train":
            self.record["performance_train"].append(performance)
            if (main_metric - self.record["best_train_main_metric"]) >= 0.001:
                self.record["best_train_main_metric"] = main_metric
                self.record["best_epoch_by_train"] = self.record["current_epoch"]
        elif update_type == "valid":
            self.record["performance_valid"].append(performance)
            if (main_metric - self.record["best_valid_main_metric"]) >= 0.001:
                self.record["best_valid_main_metric"] = main_metric
                self.record["best_epoch_by_valid"] = self.record["current_epoch"]
        elif update_type == "test":
            self.record["performance_test"].append(performance)
            if (main_metric - self.record["best_test_main_metric"]) >= 0.001:
                self.record["best_test_main_metric"] = main_metric
                self.record["best_epoch_by_test"] = self.record["current_epoch"]
        else:
            raise NotImplementedError()

    def get_performance_str(self, performance_type, index=-1):
        if performance_type == "train":
            performance = self.record["performance_train"][index]
        elif performance_type == "valid":
            performance = self.record["performance_valid"][index]
        else:
            performance = self.record["performance_test"][index]
        result = (f"main metric: {performance['main_metric']:<9.5}, AUC: {performance['AUC']:<9.5}, "
                  f"ACC: {performance['ACC']:<9.5}, RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
        return result

    def stop_training(self):
        train_strategy = self.params["train_strategy"]
        current_epoch = self.record["current_epoch"]
        best_epoch_by_valid = self.record["best_epoch_by_valid"]
        num_epoch = train_strategy["num_epoch"]
        if train_strategy["type"] == "valid_test":
            use_early_stop = train_strategy["valid_test"]["use_early_stop"]
            epoch_early_stop = train_strategy["valid_test"]["epoch_early_stop"]
            if use_early_stop:
                return (current_epoch >= num_epoch) or ((current_epoch - best_epoch_by_valid) >= epoch_early_stop)
            else:
                return current_epoch >= num_epoch
        else:
            return current_epoch >= num_epoch

    def get_evaluate_result_str(self, performance_type, performance_by):
        if performance_type == "train":
            all_performance = self.record["performance_train"]
        elif performance_type == "valid":
            all_performance = self.record["performance_valid"]
        else:
            all_performance = self.record["performance_test"]
        if performance_by == "train":
            performance_index = self.record["best_epoch_by_train"] - 1
        elif performance_by == "valid":
            performance_index = self.record["best_epoch_by_valid"] - 1
        else:
            performance_index = self.record["best_epoch_by_test"] - 1
        performance = all_performance[performance_index]
        result = (f"main metric: {performance['main_metric']:<9.5}, AUC: {performance['AUC']:<9.5}, "
                  f"ACC: {performance['ACC']:<9.5}, RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
        return result

    def get_evaluate_result(self, performance_type, performance_by):
        if performance_type == "train":
            all_performance = self.record["performance_train"]
        elif performance_type == "valid":
            all_performance = self.record["performance_valid"]
        else:
            all_performance = self.record["performance_test"]
        if performance_by == "train":
            performance_index = self.record["best_epoch_by_train"] - 1
        elif performance_by == "valid":
            performance_index = self.record["best_epoch_by_valid"] - 1
        else:
            performance_index = self.record["best_epoch_by_test"] - 1
        performance = all_performance[performance_index]
        return performance

    @staticmethod
    def cal_main_metric(performance, multi_metrics):
        """
        多指标选模型
        :param performance: {"AUC": , "ACC": , "MAE": , "RMSE": }
        :param multi_metrics: [("AUC", 1), ("ACC", 1)]
        :return:
        """
        final_metric = 0
        for metric_key, metric_weight in multi_metrics:
            if metric_key in ["AUC", "ACC"]:
                final_metric += performance[metric_key] * metric_weight
            elif metric_key in ["MAE", "RMSE"]:
                final_metric -= performance[metric_key] * metric_weight
            else:
                assert False, f"no metric named {metric_key}"

        return final_metric
