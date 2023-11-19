import torch
import random
from torch.utils.data import Dataset

from ..util.data import *
from ..util.parse import *
from .util import data_kt2srs
from .KTDataRandomAug import KTDataRandomAug
from .Similarity import OfflineSimilarity, OnlineSimilarity


class KTDataset4Aug(Dataset):
    def __init__(self, params, objects):
        super(KTDataset4Aug, self).__init__()
        self.params = params
        self.objects = objects

        self.data_uniformed = None
        self.dataset = None
        # semantic aug 所需要的
        self.semantic_pos_srs_index = []
        self.semantic_hard_neg_srs_index = []
        self.data_srs = None
        # random aug 所需要的
        self.random_data_augmentor = None
        # informative aug 所需要的
        self.offline_similarity = None
        self.online_similarity = None
        self.similarity_type = "offline"
        self.replace_aug = {
            "offline": self.informative_replace_offline,
            "online": self.informative_replace_online,
            "hybrid": self.informative_replace_hybrid
        }
        self.max_seq_len = None
        self.use_aug = True

        self.load_dataset()
        self.parse_aug()

    def set_use_aug(self):
        self.use_aug = True

    def set_not_use_aug(self):
        self.use_aug = False

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()
        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]

        if not self.use_aug:
            return result

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
            data_hard_neg = self.get_semantic_hard_neg(index)
            pad_len = max_seq_len - data_hard_neg["seq_len"]
            for k, v in data_hard_neg.items():
                if type(v) == list:
                    result[f"{k}_hard_neg"] = torch.tensor(v + [0] * pad_len).long().to(self.params["device"])
        elif aug_type == "informative_aug":
            datas_aug = self.get_informative_aug(index)
        else:
            raise NotImplementedError()

        # 补零
        for i, data_aug in enumerate(datas_aug):
            pad_len = max_seq_len - data_aug["seq_len"]
            for k, v in data_aug.items():
                if type(v) == list and k not in ["time_seq", "use_time_seq", "interval_time_seq"]:
                    # 数据增强不考虑时间
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
                    item_data_aug = KTDataRandomAug.permute_seq(item_data_aug, permute_prob, 10)
                elif aug_type == "crop":
                    item_data_aug = KTDataRandomAug.crop_seq(item_data_aug, crop_prob, 10)
                else:
                    raise NotImplementedError()
            aug_result.append(item_data_aug)
        return aug_result

    def get_semantic_aug(self, index):
        cur_same_target = deepcopy(self.semantic_pos_srs_index[index])
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
                # 因为data_srs记录的是目标习题，而做对比学习时，是用的最后一刻，所以seq len要+1
                target_seq_len = self.data_srs[pos_chosen]["target_seq_len"] + 1
                item_data_aug = deepcopy(self.data_uniformed[target_seq_id])
                for k, v in item_data_aug.items():
                    if type(v) == list:
                        item_data_aug[k] = v[:target_seq_len]
                item_data_aug["seq_len"] = target_seq_len

                delete_index = np.argwhere(cur_same_target == pos_chosen)
                cur_same_target = np.delete(cur_same_target, delete_index)
            aug_result.append(item_data_aug)
        return aug_result

    def get_semantic_hard_neg(self, index):
        cur_same_target = deepcopy(self.semantic_hard_neg_srs_index[index])
        if len(cur_same_target) == 0:
            # 没有其他人和自己做相似习题且结果不一样，用CL4KT的hard neg，即将最后一个correct取反
            item_data_hard_neg = deepcopy(self.data_uniformed[index])
            seq_len = item_data_hard_neg["seq_len"]
            last_correct = item_data_hard_neg["correct_seq"][seq_len-1]
            item_data_hard_neg["correct_seq"][seq_len-1] = int(1 - last_correct)
            for k, v in item_data_hard_neg.items():
                if type(v) == list:
                    item_data_hard_neg[k] = v[:seq_len]
            item_data_hard_neg["seq_len"] = seq_len
        else:
            hard_neg_chosen = np.random.choice(cur_same_target)
            target_seq_id = self.data_srs[hard_neg_chosen]["target_seq_id"]
            target_seq_len = self.data_srs[hard_neg_chosen]["target_seq_len"] + 1
            item_data_hard_neg = deepcopy(self.data_uniformed[target_seq_id])
            for k, v in item_data_hard_neg.items():
                if type(v) == list:
                    item_data_hard_neg[k] = v[:target_seq_len]
            item_data_hard_neg["seq_len"] = target_seq_len
        return item_data_hard_neg

    def get_informative_aug(self, index):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_aug = dataset_config_this["kt4aug"]["num_aug"]
        informative_aug_config = dataset_config_this["kt4aug"]["informative_aug"]
        aug_order = informative_aug_config["aug_order"]
        mask_prob = informative_aug_config["mask_prob"]
        insert_prob = informative_aug_config["insert_prob"]
        replace_prob = informative_aug_config["replace_prob"]
        crop_prob = informative_aug_config["crop_prob"]
        aug_result = []
        for _ in range(num_aug):
            item_data_aug = deepcopy(self.data_uniformed[index])
            seq_len = item_data_aug["seq_len"]
            for k, v in item_data_aug.items():
                if type(v) == list:
                    item_data_aug[k] = v[:seq_len]
            for aug_type in aug_order:
                if aug_type == "mask":
                    item_data_aug = self.informative_mask(item_data_aug, mask_prob, 10)
                elif aug_type == "insert":
                    item_data_aug = self.informative_insert(item_data_aug, insert_prob)
                elif aug_type == "replace":
                    item_data_aug = self.informative_replace(item_data_aug, replace_prob)
                elif aug_type == "crop":
                    item_data_aug = KTDataRandomAug.crop_seq(item_data_aug, crop_prob, 10)
                else:
                    raise NotImplementedError()
            item_data_aug["seq_len"] = len(item_data_aug["mask_seq"])
            aug_result.append(item_data_aug)
        return aug_result

    def informative_replace(self, sample, prob):
        """
        目前只考虑习题和知识点信息都有的数据集
        :param sample:
        :param prob:
        :return:
        """
        sample = deepcopy(sample)
        seq_len = sample["seq_len"]
        replace_idx = random.sample([i for i in range(seq_len)], k=max(1, int(seq_len*prob)))
        replace_func = self.replace_aug[self.similarity_type]
        for i in replace_idx:
            similar_questions = replace_func(sample["question_seq"][i])
            sample["question_seq"][i] = random.choice(similar_questions)

        return sample

    def informative_replace_offline(self, item_id, target="question", top_k=100):
        if target == "question":
            similar_items = self.offline_similarity.get_similar_question(item_id, top_k)
        else:
            similar_items = self.offline_similarity.get_similar_concept(item_id, 10)
        return similar_items

    def informative_replace_online(self, item_id, target="question", top_k=100):
        if target == "question":
            similar_items = self.offline_similarity.get_similar_question(item_id, top_k)
        else:
            similar_items = self.online_similarity.get_similar_concept(item_id, 10)
        return similar_items

    def informative_replace_hybrid(self, item_id, target="question", top_k=(3, 7)):
        if target == "question":
            similar_items = self.offline_similarity.get_similar_question(item_id, 100)
        else:
            similar_items1 = self.offline_similarity.get_similar_concept(item_id, top_k[0])
            similar_items2 = self.online_similarity.get_similar_concept(item_id, top_k[1])
            similar_items = np.concatenate((similar_items1, similar_items2))
        return similar_items

    def informative_insert(self, sample, prob):
        """
        目前只考虑习题和知识点信息都有的数据集
        :param sample:
        :param prob:
        :return:
        """
        seq_len = sample["seq_len"]
        if seq_len == self.max_seq_len:
            return sample
        seq_keys = []
        for k, v in sample.items():
            if type(v) == list:
                seq_keys.append(k)
        insert_num = max(1, min((self.max_seq_len - seq_len), int(seq_len * prob)))
        sample["seq_len"] += insert_num

        insert_idx = random.sample([i for i in range(seq_len - 1)], k=insert_num)
        do_insert = [False for _ in range(seq_len)]
        for i in insert_idx:
            do_insert[i] = True

        sample_new = {k: [] for k in seq_keys}
        replace_func = self.replace_aug[self.similarity_type]
        for i, insert_flag in enumerate(do_insert):
            for k in seq_keys:
                sample_new[k].append(sample[k][i])
            if insert_flag:
                similar_concepts = replace_func(sample["concept_seq"][i], target="concept")
                similar_concept = random.choice(similar_concepts)
                sample_new["concept_seq"].append(similar_concept)
                # 当前做错了，则插入的interaction也为做错，否则有1/2概率作对
                if sample["correct_seq"][i] == 0:
                    sample_new["correct_seq"].append(0)
                else:
                    sample_new["correct_seq"].append(random.choice([0, 1]))
                sample_new["question_seq"].append(self.offline_similarity.get_random_q_in_concept(similar_concept))
                sample_new["mask_seq"].append(1)
        return sample_new

    def informative_mask(self, sample, mask_prob, mask_min_seq_len=10):
        seq_len = sample["seq_len"]
        if seq_len < mask_min_seq_len:
            return sample
        seq_keys = []
        for k, v in sample.items():
            if type(v) == list:
                seq_keys.append(k)
        sample_new = {k: [] for k in seq_keys}
        replace_func = self.replace_aug[self.similarity_type]
        for i in range(seq_len):
            prob = random.random()
            condition1 = prob < mask_prob
            condition2 = i < (seq_len - 1)
            # 相对无关的知识点可以被mask，即如果当前知识点和后一个知识点关联较大，则不能mask
            condition3 = condition2 and sample["concept_seq"][i+1] not in replace_func(sample["concept_seq"][i], target="concept")
            if condition1 and condition2 and condition3:
                continue
            for k in seq_keys:
                sample_new[k].append(sample[k][i])
        sample_new["seq_len"] = len(sample_new["correct_seq"])

        return sample_new

    def set_sim_type(self, sim_type):
        if sim_type in ["offline", "online", "hybrid"]:
            self.similarity_type = sim_type
        else:
            raise NotImplementedError()

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
                    for time_j in range(1, len(item_data["time_seq"])):
                        interval_time = (item_data["time_seq"][time_j] - item_data["time_seq"][time_j - 1]) // 60
                        interval_time = max(0, min(interval_time, 60 * 24 * 30))
                        interval_time_seq.append(interval_time)
                    dataset_converted["interval_time_seq"].append(interval_time_seq)
                else:
                    dataset_converted[k].append(item_data[k])
            dataset_converted["seq_id"].append(seq_i)
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
        elif aug_type == "informative_aug":
            self.informative_parse()
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
            self.semantic_pos_srs_index = semantic_aug[0]
            self.semantic_hard_neg_srs_index = semantic_aug[1]
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
                target_correct = data_srs[idx]["target_correct"]
                if target_seq_id == i:
                    index2delete.append(k)
                if (target_seq_id != i) and (target_correct != last_correct):
                    index2delete.append(k)
                    index2hard_neg.append(idx)

            all_index_same_id_wo_self = np.delete(all_index_same_q, index2delete)
            self.semantic_pos_srs_index.append(all_index_same_id_wo_self)
            self.semantic_hard_neg_srs_index.append(np.array(index2hard_neg))

        self.semantic_pos_srs_index = np.array(self.semantic_pos_srs_index, dtype=object)
        self.semantic_hard_neg_srs_index = np.array(self.semantic_hard_neg_srs_index, dtype=object)
        semantic_aug = np.array([self.semantic_pos_srs_index, self.semantic_hard_neg_srs_index], dtype=object)
        np.save(semantic_aug_path, semantic_aug)

    def get_statics_kt_dataset(self):
        num_seq = len(self.dataset["mask_seq"])
        with torch.no_grad():
            num_sample = torch.sum(self.dataset["mask_seq"][:, 1:]).item()
            num_interaction = torch.sum(self.dataset["mask_seq"]).item()
            correct_seq = self.dataset["correct_seq"]
            mask_bool_seq = torch.ne(self.dataset["mask_seq"], 0)
            num_correct = torch.sum(torch.masked_select(correct_seq, mask_bool_seq)).item()
        return num_seq, num_sample, num_correct / num_interaction

    def informative_parse(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        data_type = self.params["datasets_config"]["data_type"]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)

        if dataset_path != "":
            data = read_preprocessed_file(dataset_path)
        else:
            data = self.objects["dataset_this"]
        self.max_seq_len = len(data[0]["mask_seq"])
        data = data_util.dataset_delete_pad(data)
        self.offline_similarity = OfflineSimilarity(self.params, self.objects)
        self.offline_similarity.parse(data, data_type)
        self.online_similarity = OnlineSimilarity()

    def update_online_sim4info_aug(self):
        pass
