import random

import numpy as np
import torch
from torch.utils.data import Dataset

from ..util.data import *
from ..util.parse import *
from .util import data_kt2srs
from .KTDataRandomAug import KTDataRandomAug


class KTDataset4Aug(Dataset):
    def __init__(self, params, objects):
        super(KTDataset4Aug, self).__init__()
        self.params = params
        self.objects = objects

        self.data_uniformed = None
        self.dataset = None
        # semantic aug 所需要的
        self.semantic_pos_seq_id = []
        self.semantic_pos_index = []
        self.semantic_hard_neg_seq_id = []
        self.semantic_hard_neg_index = []
        self.data_srs = None
        # random aug 所需要的
        self.random_data_augmentor = None

        self.load_dataset()
        self.parse_aug()

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()
        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]
        max_seq_len = result["mask_seq"].shape[0]

        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        aug_type = dataset_config_this["kt4aug"]["aug_type"]
        if aug_type == "random_aug":
            random_aug_config = dataset_config_this["kt4aug"]["random_aug"]
            datas_aug = self.get_random_aug(index)
            hard_neg_prob = random_aug_config["hard_neg_prob"]
            seq_len = self.data_uniformed[index]["seq_len"]
            correct_seq_neg = KTDataRandomAug.negative_seq(self.data_uniformed[index]["correct_seq"][:seq_len], hard_neg_prob)
            result["correct_seq_hard_neg"] = (
                torch.tensor(correct_seq_neg + [0] * (max_seq_len - seq_len)).long().to(self.params["device"]))
        elif aug_type == "semantic_aug":
            datas_aug = self.get_semantic_aug(index)
        else:
            raise NotImplementedError()

        # 补零
        for i, data_aug in enumerate(datas_aug):
            pad_len = max_seq_len - data_aug["seq_len"]
            for k, v in data_aug.items():
                if type(v) == list:
                    result[f"{k}_aug_{i}"] = torch.tensor(v + [0] * pad_len).long().to(self.params["device"])

        return result

    def get_random_aug(self, index):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_aug = dataset_config_this["kt4aug"]["num_aug"]
        random_aug_config = dataset_config_this["kt4aug"]["random_aug"]
        aug_order = random_aug_config["aug_order"]
        mask_prob = random_aug_config["mask_prob"]
        replace_prob = random_aug_config["replace_prob"]
        permute_prob = random_aug_config["permute_prob"]
        crop_prob = random_aug_config["crop_prob"]
        aug_result = []
        for _ in range(num_aug):
            item_data_aug = deepcopy(self.data_uniformed[index])
            seq_len = item_data_aug["seq_len"]
            for k, v in item_data_aug.items():
                if type(v) == list:
                    item_data_aug[k] = v[:seq_len]
            for aug_type in aug_order:
                if aug_type == "mask":
                    item_data_aug = KTDataRandomAug.mask_seq(item_data_aug, mask_prob, 10)
                elif aug_type == "replace":
                    item_data_aug = self.random_data_augmentor.replace_seq(item_data_aug, replace_prob)
                elif aug_type == "permute":
                    KTDataRandomAug.permute_seq(item_data_aug, permute_prob, 10)
                elif aug_type == "crop":
                    KTDataRandomAug.crop_seq(item_data_aug, crop_prob, 10)
                else:
                    raise NotImplementedError()
            aug_result.append(item_data_aug)
        return aug_result

    def get_semantic_aug(self, index):
        cur_same_target = deepcopy(self.semantic_pos_seq_id[index])
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_aug = dataset_config_this["kt4aug"]["num_aug"]
        aug_result = []
        for _ in range(num_aug):
            if len(cur_same_target) == 0:
                # 没有其他人和自己做相似习题且结果一样，或者随机增强样本的数量超过相似样本数量，就用随机增强代替
                item_data_aug = deepcopy(self.data_uniformed[index])
                seq_len = item_data_aug["seq_len"]
                for k, v in item_data_aug.items():
                    if type(v) == list:
                        item_data_aug[k] = v[:seq_len]
                item_data_aug = KTDataRandomAug.mask_seq(item_data_aug, 0.2, 10)
                item_data_aug = KTDataRandomAug.crop_seq(item_data_aug, 0.1, 10)
                item_data_aug = KTDataRandomAug.permute_seq(item_data_aug, 0.2, 10)
            else:
                pos_chosen = np.random.choice(cur_same_target)
                target_seq_id = self.data_srs[pos_chosen]["target_seq_id"]
                target_seq_len = self.data_srs[pos_chosen]["target_seq_len"]
                item_data_aug = deepcopy(self.data_uniformed[target_seq_id])
                for k, v in item_data_aug.items():
                    if type(v) == list:
                        item_data_aug[k] = v[:target_seq_len]
                item_data_aug["seq_len"] = target_seq_len

                delete_index = np.argwhere(cur_same_target == pos_chosen)
                cur_same_target = np.delete(cur_same_target, delete_index)
            aug_result.append(item_data_aug)
        return aug_result

    def load_dataset(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        data_type = self.params["datasets_config"]["data_type"]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        unuseful_keys = dataset_config_this["unuseful_seq_keys"]
        unuseful_keys = unuseful_keys - {"seq_len"}

        if dataset_path != "":
            dataset_original = read_preprocessed_file(dataset_path)
        else:
            dataset_original = self.objects["dataset_this"]
        if data_type == "multi_concept":
            self.data_uniformed = data_agg_question(dataset_original)
        else:
            self.data_uniformed = deepcopy(dataset_original)

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
        for i, item_data in enumerate(dataset_original):
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
                    for i in range(1, len(item_data["time_seq"])):
                        interval_time = (item_data["time_seq"][i] - item_data["time_seq"][i - 1]) // 60
                        interval_time = max(0, min(interval_time, 60 * 24 * 30))
                        interval_time_seq.append(interval_time)
                    dataset_converted["interval_time_seq"].append(interval_time_seq)
                else:
                    dataset_converted[k].append(item_data[k])
            dataset_converted["seq_id"].append(i)
        if "time_seq" in dataset_converted.keys():
            del dataset_converted["time_seq"]
        if "question_seq_mask" in dataset_converted.keys():
            del dataset_converted["question_seq_mask"]

        for k in dataset_converted.keys():
            dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.params["device"])
        self.dataset = dataset_converted

    def parse_aug(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        aug_type = dataset_config_this["kt4aug"]["aug_type"]
        if aug_type == "semantic_aug":
            self.semantic_parse()
        elif aug_type == "random_aug":
            self.random_data_augmentor = KTDataRandomAug(self.params, self.objects)
            self.random_data_augmentor.parse_data(self.data_uniformed)
        else:
            raise NotImplementedError()

    def semantic_parse(self):
        """
        解析数据集，在相同习题上有相同结果的作为正样本，不同结果的作为hard neg
        :return:
        """
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        data_type = self.params["datasets_config"]["data_type"]
        data_srs = data_kt2srs(self.data_uniformed, data_type)
        self.data_srs = data_srs
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        setting_dir = self.objects["file_manager"].get_setting_dir(setting_name)
        semantic_aug_name = file_name.replace(".txt", "_semantic_augmentation.npy")
        semantic_aug_path = os.path.join(setting_dir, semantic_aug_name)
        if os.path.exists(semantic_aug_path):
            semantic_aug = np.load(semantic_aug_path, allow_pickle=True)
            self.semantic_pos_seq_id = semantic_aug[0]
            self.semantic_pos_index = semantic_aug[1]
            self.semantic_hard_neg_seq_id = semantic_aug[2]
            self.semantic_hard_neg_index = semantic_aug[3]
            return

        target_q_all = np.array(list(map(lambda x: x["target_question"], data_srs)))
        for i, item_data in enumerate(self.data_uniformed):
            last_q = item_data["question_seq"][item_data["seq_len"]-1]
            last_correct = item_data["correct_seq"][item_data["seq_len"]-1]
            all_index_same_q = np.where(target_q_all == last_q)[0]
            index2delete = []
            index2hard_neg = []
            for k, idx in enumerate(all_index_same_q):
                target_seq_id = data_srs[idx]["target_seq_id"]
                target_seq_len = data_srs[idx]["target_seq_len"]
                target_correct = data_srs[idx]["target_correct"]
                if target_seq_id == i and target_seq_len == (item_data["seq_len"]-1):
                    index2delete.append(k)
                if target_correct != last_correct:
                    index2delete.append(k)
                    index2hard_neg.append(k)

            all_index_same_id_wo_self = np.delete(all_index_same_q, index2delete)
            self.semantic_pos_seq_id.append(
                list(map(lambda idx_sam_q: data_srs[idx_sam_q]["target_seq_id"], all_index_same_id_wo_self)))
            # 因为data_srs记录的是目标习题，而做对比学习时，是用的最后一刻，所以seq len要+1
            self.semantic_pos_index.append(
                list(map(lambda idx_sam_q: data_srs[idx_sam_q]["target_seq_len"] + 1, all_index_same_id_wo_self)))

            self.semantic_hard_neg_seq_id.append(
                list(map(lambda idx_sam_q: data_srs[idx_sam_q]["target_seq_id"], index2hard_neg)))
            self.semantic_hard_neg_index.append(
                list(map(lambda idx_sam_q: data_srs[idx_sam_q]["target_seq_len"] + 1, index2hard_neg)))
        self.semantic_pos_seq_id = np.array(self.semantic_pos_seq_id, dtype=object)
        self.semantic_pos_index = np.array(self.semantic_pos_index, dtype=object)
        self.semantic_hard_neg_seq_id = np.array(self.semantic_hard_neg_seq_id, dtype=object)
        self.semantic_hard_neg_index = np.array(self.semantic_hard_neg_index, dtype=object)
        semantic_aug = np.array([self.semantic_pos_seq_id, self.semantic_pos_index,
                                 self.semantic_hard_neg_seq_id, self.semantic_hard_neg_index], dtype=object)
        np.save(semantic_aug_path, semantic_aug)
