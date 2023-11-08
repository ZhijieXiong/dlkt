class TrainRecord:
    def __init__(self):
        self.record = {
            "current_epoch": 1,
            "best_epoch_by_valid": 1,
            "best_epoch_by_test": 1,
            "performance_valid": [],
            "performance_test": []
        }
        self.objects = {}

    def next_epoch(self):
        self.record["current_epoch"] += 1

    def update_record_valid_test(self, valid_performance, test_performance):
        current_best_valid_metric = self.record["performance_valid"][self.record["best_epoch_by_valid"]]["AUC"]
        if (valid_performance["AUC"] - current_best_valid_metric) > 0.0001:
            pass

    def cal_main_metric(self):
        pass

