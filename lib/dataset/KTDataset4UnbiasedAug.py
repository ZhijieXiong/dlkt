import random
import torch
import os

from copy import deepcopy
from torch.utils.data import Dataset

from ..util.data import read_preprocessed_file, dataset_delete_pad
from ..util.parse import get_keys_from_uniform


# 该data loader不考虑multi concept数据格式
class KTDataset4UnbiasedAug(Dataset):
    def __init__(self, params, objects):
        super(KTDataset4UnbiasedAug, self).__init__()
        self.params = params
        self.objects = objects

        self.data_uniformed = None
        self.dataset = None
        self.load_dataset()

    def load_dataset(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
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
        seq_keys = list(set(seq_keys) - unuseful_keys - {"age_seq"})
        unuseful_keys = all_keys - set(id_keys).union(set(seq_keys))

        if "time_seq" in seq_keys:
            for item_data in dataset_original:
                item_data["interval_time_seq"] = [0]
                for time_j in range(1, len(item_data["time_seq"])):
                    interval_time = (item_data["time_seq"][time_j] - item_data["time_seq"][time_j - 1]) // 60
                    interval_time = max(0, min(interval_time, 60 * 24 * 30))
                    item_data["interval_time_seq"].append(interval_time)
                del item_data["time_seq"]
            seq_keys = set(seq_keys) - {"time_seq"}
            seq_keys.add("interval_time_seq")
            seq_keys = list(seq_keys)
        for item_data in dataset_original:
            for k in unuseful_keys:
                del item_data[k]
        self.data_uniformed = dataset_delete_pad(dataset_original)

        dataset_converted = {k: [] for k in (id_keys + seq_keys)}
        dataset_converted["seq_id"] = []
        for seq_i, item_data in enumerate(dataset_original):
            for k in id_keys:
                dataset_converted[k].append(item_data[k])
            for k in seq_keys:
                dataset_converted[k].append(item_data[k])
            dataset_converted["seq_id"].append(seq_i)

        for k in dataset_converted.keys():
            if k not in ["weight_seq", "hint_factor_seq", "attempt_factor_seq", "time_factor_seq", "correct_float"]:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.params["device"])
            else:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.params["device"])
        self.dataset = dataset_converted

    def get_unbiased_aug(self, item_data2aug):
        aug_config = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]["kt4unbiased_aug"]
        num_aug = aug_config["num_aug"]
        unbias_intensity = aug_config["unbias_intensity"]
        num_question = aug_config["num_question"]
        num_concept = aug_config["num_concept_single_concept"]
        question2concept = self.objects["unbias"]["question2concept_single_concept"]

        seq_len = item_data2aug["seq_len"]
        num_item2unbias = aug_config["num_item2unbias"]
        if num_item2unbias >= seq_len:
            num_item2unbias = seq_len - 1
        aug_result = []
        for _ in range(num_aug):
            item_data_aug = deepcopy(item_data2aug)
            correct_seq_latest = item_data_aug["correct_seq"][-num_item2unbias-1:]
            acc_latest = sum(correct_seq_latest) / len(correct_seq_latest)
            if acc_latest > 0.5:
                correct2unbias = 1
                unbias_intensity_bound = acc_latest - 0.5
            elif acc_latest < 0.5:
                correct2unbias = 0
                unbias_intensity_bound = 0.5 - acc_latest
            else:
                correct2unbias = random.randint(0, 1)
                unbias_intensity_bound = unbias_intensity

            if unbias_intensity_bound > unbias_intensity:
                unbias_ratio = random.random() * (unbias_intensity_bound - unbias_intensity) + unbias_intensity
            elif unbias_intensity_bound < unbias_intensity:
                unbias_ratio = random.random() * (unbias_intensity - unbias_intensity_bound) + unbias_intensity_bound
            else:
                unbias_ratio = 0.05

            # 将序列分成两段（按照num_item2unbias分成最近一段和前面一段，前面一段保持不变）
            item_data_last = {
               k: v if (type(v) is not list) else v[-num_item2unbias-1:] for k, v in item_data_aug.items()
            }
            for k, v in item_data_aug.items():
                if type(v) is list:
                    item_data_aug[k] = v[:-num_item2unbias-1]

            # 按照unbias_ratio随机选择添加习题交互，以平衡序列正确率接近0.5
            for i, correct in enumerate(correct_seq_latest[:-1]):
                if (correct == correct2unbias) and (random.random() < unbias_ratio):
                    # 添加增强数据
                    current_q = item_data_last["question_seq"][i]
                    correspond_c = question2concept[current_q][0]
                    for k, v in item_data_aug.items():
                        if k == "question_seq":
                            # 添加一道virtual question（添加到embedd question最后面），每个知识点（如果是multi concept数据集，
                            # 则为其single concept格式的知识点）各两道，分别表示该知识点（组合）下最难和最简单的题
                            item_data_aug["question_seq"].append(num_question + correspond_c + num_concept * correct)
                        elif k == "correct_seq":
                            item_data_aug["correct_seq"].append(1 - correct2unbias)
                        elif k == "interval_time_seq":
                            item_data_aug["interval_time_seq"].append(0)
                        elif type(v) is list:
                            item_data_aug[k].append(item_data_last[k][i])
                # 添加原数据
                for k, v in item_data_aug.items():
                    if type(v) is list:
                        item_data_aug[k].append(item_data_last[k][i])

            # 添加最后一时刻数据，即target
            for k, v in item_data_aug.items():
                if type(v) is list:
                    item_data_aug[k].append(item_data_last[k][-1])
            item_data_aug["seq_len"] = len(item_data_aug["mask_seq"])
            aug_result.append(item_data_aug)

        return aug_result

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()

        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]

        max_seq_len = result["mask_seq"].shape[0]
        item_data2aug = deepcopy(self.data_uniformed[index])
        seq_len = item_data2aug["seq_len"]
        if seq_len > 10:
            seq_len = random.randint(10, seq_len)
            for k, v in item_data2aug.items():
                if type(v) == list:
                    item_data2aug[k] = v[:seq_len]
        item_data2aug["seq_len"] = seq_len
        datas_aug = self.get_unbiased_aug(item_data2aug)

        # 补零
        for i, data_aug in enumerate(datas_aug):
            pad_len = max_seq_len - data_aug["seq_len"]
            for k, v in data_aug.items():
                if type(v) == list:
                    result[f"{k}_aug_{i}"] = torch.tensor(v + [0] * pad_len).long().to(self.params["device"])

        return result
