import math
import random
import torch
import os

from copy import deepcopy
from torch.utils.data import Dataset

from ..CONSTANT import INTERVAL_TIME4LPKT_PLUS, USE_TIME4LPKT_PLUS
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

        max_seq_len = len(dataset_original[0]["mask_seq"])
        if "time_seq" in seq_keys:
            for item_data in dataset_original:
                item_data["interval_time_seq"] = [0]
                seq_len = item_data["seq_len"]
                for time_j in range(1, seq_len):
                    interval_time_real = (item_data["time_seq"][time_j] - item_data["time_seq"][time_j - 1]) // 60
                    interval_time_idx = len(INTERVAL_TIME4LPKT_PLUS)
                    for idx, interval_time_value in enumerate(INTERVAL_TIME4LPKT_PLUS):
                        if interval_time_real < 0:
                            interval_time_idx = 0
                            break
                        if interval_time_real <= interval_time_value:
                            interval_time_idx = idx
                            break
                    item_data["interval_time_seq"].append(interval_time_idx)
                    item_data["interval_time_seq"] += [0] * (max_seq_len - seq_len)
                del item_data["time_seq"]
            seq_keys = set(seq_keys) - {"time_seq"}
            seq_keys.add("interval_time_seq")
            seq_keys = list(seq_keys)
        if "use_time_seq" in seq_keys:
            for item_data in dataset_original:
                seq_len = item_data["seq_len"]
                for time_i, use_time in enumerate(item_data["use_time_seq"][:seq_len]):
                    use_time_idx = len(USE_TIME4LPKT_PLUS)
                    for idx, use_time_value in enumerate(USE_TIME4LPKT_PLUS):
                        if use_time <= use_time_value:
                            use_time_idx = idx
                            break
                    item_data["use_time_seq"][time_i] = use_time_idx

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

    def get_unbiased_aug(self, item_data2aug, max_seq_len=200):
        aug_config = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]["kt4aug"]
        num_aug = aug_config["num_aug"]
        unbiased_aug_config = aug_config["unbiased_aug"]
        num_item2unbias = unbiased_aug_config["num_item2unbias"]
        num_question = unbiased_aug_config["num_question"]
        num_concept = unbiased_aug_config["num_concept_single_concept"]
        use_virtual_emb4question = unbiased_aug_config["use_virtual_emb4question"]
        use_virtual_emb4aux = unbiased_aug_config["use_virtual_emb4aux"]
        question2concept = self.objects["unbiased_aug"]["question2concept_single_concept"]
        most_question = self.objects["unbiased_aug"]["most_question"]

        assert 5 <= num_item2unbias <= (max_seq_len // 2), f"{num_item2unbias} must be in [5, max_seq_len // 2]"
        if num_item2unbias >= item_data2aug["seq_len"]:
            num_item2unbias = item_data2aug["seq_len"] - 1
        aug_result = []
        for _ in range(num_aug):
            item_data_aug = deepcopy(item_data2aug)
            correct_seq = item_data_aug["correct_seq"]
            correct_seq_latest = correct_seq[-num_item2unbias-1:]
            acc_latest = sum(correct_seq_latest) / len(correct_seq_latest)
            if acc_latest > 0.5:
                correct2unbias = 1
                unbias_ratio = acc_latest - 0.5
            elif acc_latest < 0.5:
                correct2unbias = 0
                unbias_ratio = 0.5 - acc_latest
            else:
                # 如果本身就是平衡的，则添加一些微弱的随机扰动
                correct2unbias = random.randint(0, 1)
                unbias_ratio = 0.05

            # 将序列分成两段（按照num_item2unbias分成最近一段和前面一段，前面一段保持不变）
            item_data_last = {
               k: v if (type(v) is not list) else v[-num_item2unbias-1:] for k, v in item_data_aug.items()
            }
            item_data_last["seq_len"] = len(item_data_last["mask_seq"])
            for k, v in item_data_aug.items():
                if type(v) is list:
                    item_data_aug[k] = v[:-num_item2unbias-1]
            item_data_aug["seq_len"] = len(item_data_aug["mask_seq"])

            # 按照unbias_ratio随机选择添加习题交互，以平衡序列正确率接近0.5
            num_added_item = 0
            num2add_item = max(1, min(num_item2unbias, math.ceil(num_item2unbias * unbias_ratio)))
            add_origin = False
            while num_added_item < 1:
                # 至少添加1个
                for i, correct in enumerate(correct_seq_latest[:-1]):
                    if (correct == correct2unbias) and (random.random() < unbias_ratio) and (num_added_item <= num2add_item):
                        # 添加增强数据
                        num_added_item += 1
                        current_q = item_data_last["question_seq"][i]
                        correspond_c = question2concept[current_q][0]
                        for k, v in item_data_aug.items():
                            if k == "question_seq":
                                if use_virtual_emb4question:
                                    # 使用virtual question（添加到embedd question最后面），每个知识点下最难和最简单的题
                                    question2add = num_question + correspond_c + num_concept * correct
                                else:
                                    # 根据统计信息对应知识点组合下最难和最简单的习题
                                    question2add = most_question[correspond_c]["most_hard"] \
                                        if correct2unbias == 1 else most_question[correspond_c]["most_easy"]
                                item_data_aug["question_seq"].append(question2add)
                            elif k == "correct_seq":
                                item_data_aug["correct_seq"].append(1 - correct2unbias)
                            elif k == "interval_time_seq":
                                item_data_aug["interval_time_seq"].append(0)
                            elif k == "num_hint_seq":
                                if use_virtual_emb4aux:
                                    num_hint = 100 + correct2unbias
                                else:
                                    num_hint = 5 if correct2unbias else 0
                                item_data_aug["num_hint_seq"].append(num_hint)
                            elif k == "num_attempt_seq":
                                if use_virtual_emb4aux:
                                    num_attempt = 100 + correct2unbias
                                else:
                                    num_attempt = 5 if correct2unbias else 0
                                item_data_aug["num_attempt_seq"].append(num_attempt)
                            elif k == "use_time_seq":
                                if use_virtual_emb4aux:
                                    use_time = 100 + correct2unbias
                                else:
                                    use_time = 5 if correct2unbias else 0
                                item_data_aug["use_time_seq"].append(use_time)
                            elif k == "use_time_first_seq":
                                if use_virtual_emb4aux:
                                    use_time_first = 100 + correct2unbias
                                else:
                                    use_time_first = 5 if correct2unbias else 0
                                item_data_aug["use_time_first_seq"].append(use_time_first)
                            elif k == "mask_seq":
                                item_data_aug["mask_seq"].append(1)
                            elif type(v) is list:
                                item_data_aug[k].append(item_data_last[k][i])
                    if not add_origin:
                        # 添加原数据
                        for k, v in item_data_aug.items():
                            if type(v) is list:
                                item_data_aug[k].append(item_data_last[k][i])
                # 防止原始数据重复添加
                add_origin = True

            # 添加最后一时刻数据，即target
            for k, v in item_data_aug.items():
                if type(v) is list:
                    item_data_aug[k].append(item_data_last[k][-1])
            item_data_aug["seq_len"] = len(item_data_aug["mask_seq"])

            # 防止超过最大长度
            if item_data_aug["seq_len"] > max_seq_len:
                item_data_aug["seq_len"] = max_seq_len
                for k, v in item_data_aug.items():
                    if type(v) is list:
                        item_data_aug[k] = v[-max_seq_len:]

            aug_result.append(item_data_aug)

        return aug_result

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()

        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]

        num_aug = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]["kt4aug"]["num_aug"]
        if num_aug == 0:
            return result

        max_seq_len = result["mask_seq"].shape[0]
        item_data2aug = deepcopy(self.data_uniformed[index])
        seq_len = item_data2aug["seq_len"]
        if seq_len > 10:
            seq_len = random.randint(10, seq_len)
            for k, v in item_data2aug.items():
                if type(v) == list:
                    item_data2aug[k] = v[:seq_len]
        result["seq_len_original"] = torch.tensor(seq_len).long().to(self.params["device"])
        item_data2aug["seq_len"] = seq_len
        datas_aug = self.get_unbiased_aug(item_data2aug, max_seq_len)

        # 补零
        for i, data_aug in enumerate(datas_aug):
            pad_len = max_seq_len - data_aug["seq_len"]
            result[f"seq_len_aug_{i}"] = torch.tensor(data_aug["seq_len"]).long().to(self.params["device"])
            for k, v in data_aug.items():
                if type(v) == list:
                    result[f"{k}_aug_{i}"] = torch.tensor(v + [0] * pad_len).long().to(self.params["device"])

        return result

    def get_statics_kt_dataset(self):
        num_seq = len(self.dataset["mask_seq"])
        with torch.no_grad():
            num_sample = torch.sum(self.dataset["mask_seq"][:, 1:]).item()
            num_interaction = torch.sum(self.dataset["mask_seq"]).item()
            correct_seq = self.dataset["correct_seq"]
            mask_bool_seq = torch.ne(self.dataset["mask_seq"], 0)
            num_correct = torch.sum(torch.masked_select(correct_seq, mask_bool_seq)).item()
        return num_seq, num_sample, num_correct / num_interaction
