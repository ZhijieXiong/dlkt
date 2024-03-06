import torch
from torch.utils.data import Dataset

from ..util.data import *
from ..util.parse import *


class KTDataset4AT_DKT(Dataset):
    def __init__(self, params, objects):
        super(KTDataset4AT_DKT, self).__init__()
        self.params = params
        self.objects = objects
        self.dataset = None

        self.load_dataset()

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()
        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]
        return result

    def load_dataset(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        data_type = self.params["datasets_config"]["data_type"]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        unuseful_keys = dataset_config_this.get("unuseful_seq_keys", {"user_id"})
        unuseful_keys = unuseful_keys - {"seq_len"}

        if dataset_path != "":
            dataset_original = read_preprocessed_file(dataset_path)
        else:
            dataset_original = self.objects["dataset_this"]

        id_keys, seq_keys = get_keys_from_uniform(dataset_original)
        all_keys = set(id_keys).union(seq_keys)
        id_keys = list(set(id_keys) - unuseful_keys)
        seq_keys = list(set(seq_keys) - unuseful_keys)
        unuseful_keys = all_keys - set(id_keys).union(set(seq_keys))
        for item_data in dataset_original:
            for k in unuseful_keys:
                del item_data[k]

        dataset_converted = {k: [] for k in (id_keys + seq_keys)}
        if "question_seq" in seq_keys:
            dataset_converted["question_seq_mask"] = []
        if "time_seq" in seq_keys:
            dataset_converted["interval_time_seq"] = []
        dataset_converted["seq_id"] = []
        dataset_converted["history_acc_seq"] = []
        for seq_i, item_data in enumerate(dataset_original):
            for k in id_keys:
                dataset_converted[k].append(item_data[k])
            for k in seq_keys:
                if data_type == "multi_concept" and k == "question_seq":
                    question_seq = item_data["question_seq"]
                    question_seq_new = []
                    current_q = question_seq[0]
                    for q in question_seq:
                        if q != -1:
                            current_q = q
                        question_seq_new.append(current_q)
                    dataset_converted["question_seq"].append(question_seq_new)
                    dataset_converted["question_seq_mask"].append(question_seq)
                elif k == "time_seq":
                    interval_time_seq = [0]
                    for time_i in range(1, len(item_data["time_seq"])):
                        interval_time = (item_data["time_seq"][time_i] - item_data["time_seq"][time_i - 1]) // 60
                        interval_time = max(0, min(interval_time, 60 * 24 * 30))
                        interval_time_seq.append(interval_time)
                    dataset_converted["interval_time_seq"].append(interval_time_seq)
                else:
                    dataset_converted[k].append(item_data[k])

            history_acc_seq = []
            right, total = 0, 0
            for correct in item_data["correct_seq"]:
                if correct:
                    right += 1
                total += 1
                history_acc_seq.append(right / total)
            dataset_converted["history_acc_seq"].append(history_acc_seq)

            dataset_converted["seq_id"].append(seq_i)
        if "time_seq" in dataset_converted.keys():
            del dataset_converted["time_seq"]
        if "question_seq_mask" in dataset_converted.keys():
            del dataset_converted["question_seq_mask"]

        for k in dataset_converted.keys():
            if k == "history_acc_seq":
                dataset_converted[k] = torch.tensor(dataset_converted[k]).float().to(self.params["device"])
            else:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.params["device"])
        self.dataset = dataset_converted

    def get_statics_kt_dataset(self):
        num_seq = len(self.dataset["mask_seq"])
        with torch.no_grad():
            num_sample = torch.sum(self.dataset["mask_seq"][:, 1:]).item()
            num_interaction = torch.sum(self.dataset["mask_seq"]).item()
            correct_seq = self.dataset["correct_seq"]
            mask_bool_seq = torch.ne(self.dataset["mask_seq"], 0)
            num_correct = torch.sum(torch.masked_select(correct_seq, mask_bool_seq)).item()
        return num_seq, num_sample, num_correct / num_interaction
