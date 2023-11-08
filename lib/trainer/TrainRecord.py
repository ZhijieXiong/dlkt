class TrainRecord:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.record = {
            "current_epoch": 1,
            "best_valid_main_metric": 0,
            "best_test_main_metric": 0,
            "best_epoch_by_valid": 1,
            "best_epoch_by_test": 1,
            "performance_valid": [],
            "performance_test": []
        }

    def next_epoch(self, test_performance, valid_performance=None):
        train_strategy = self.params["train_strategy"]
        self.record["current_epoch"] += 1
        if train_strategy["type"] == "valid_test":
            self.update_best_metric(valid_performance, update_type="valid")
        self.update_best_metric(test_performance, update_type="test")

    def update_best_metric(self, performance, update_type):
        use_mutil_metric = self.params["train_strategy"]["valid_test"]["use_mutil_metric"]
        main_metric_key = self.params["train_strategy"]["valid_test"]["main_metric"]
        mutil_metrics = self.params["train_strategy"]["valid_test"]["multi_metrics"]
        if not use_mutil_metric:
            main_metric = self.cal_main_metric(performance, mutil_metrics)
        else:
            main_metric = self.params["train_strategy"][main_metric_key]

        if update_type == "valid":
            if (main_metric - self.record["best_valid_main_metric"]) > 0.0001:
                self.record["best_valid_main_metric"] = main_metric
                self.record["best_epoch_by_valid"] = self.record["current_epoch"]
        elif update_type == "test":
            if (main_metric - self.record["best_test_main_metric"]) > 0.0001:
                self.record["best_test_main_metric"] = main_metric
                self.record["best_epoch_by_test"] = self.record["current_epoch"]
        else:
            raise NotImplementedError()

    def get_performance_str(self, performance_type):
        if performance_type == "valid":
            pass
        else:
            pass

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
