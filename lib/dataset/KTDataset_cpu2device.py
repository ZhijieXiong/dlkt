import random

import torch
from torch.utils.data import Dataset

from ..util.data import *
from ..util.parse import *


# 做对easy题，那么做对更简单的easy题的可信度为0.5，做错hard题的可信度为0.25
# 做错easy题，那么做对更简单的easy题的可信度为0.25，做错hard题的可信度为1
# 做对hard题，那么做对easy题的可信度为1，做错更难的hard题的可信度为0.25
# 做错hard题，那么做对easy题的可信度为0.25，做错更难的hard题的可信度为0.5
# 以easy为例，(做错该题：(做对更简单题的可信度, 做错更难题的可信度), 做对该题：(做对更简单题的可信度, 做错更难题的可信度))
WEIGHT_TABLE = {
    "unknown": ((0.125, 0.125), (0.125, 0.125)),
    "easy": ((0.25, 1), (0.5, 0.25)),
    "hard": ((0.25, 0.5), (1, 0.25)),
    "middle": ((0.25, 0.25), (0.25, 0.25)),
}


class KTDataset_cpu2device(Dataset):
    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects
        self.dataset = None

        self.load_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        result = dict()
        item_data = self.dataset[index]

        for k in item_data.keys():
            result[k] = torch.tensor(item_data[k]).long().to(self.params["device"])

        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        data_type = self.params["datasets_config"]["data_type"]
        kt_dataset_type = dataset_config_this["type"]

        if kt_dataset_type == "kt_enhance":
            max_seq_len = len(item_data["mask_seq"])
            pad_len = max_seq_len - item_data["seq_len"]
            concept_dict = self.objects["kt_enhance"]["concept_dict"]
            question_dict = self.objects["kt_enhance"]["question_dict"]
            question_easier_seq = []
            question_harder_seq = []
            concept_easier_seq = []
            concept_harder_seq = []
            weight_easier_seq = []
            weight_harder_seq = []
            mask_easier_seq = []
            mask_harder_seq = []
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                correct = item_data["correct_seq"][i]
                if data_type == "single_concept":
                    c_pair = question_dict[q_id][1][0]
                    c_id = c_pair[0]
                    q_diff = c_pair[1]
                    k = c_pair[2]
                    weight_easier = WEIGHT_TABLE[q_diff][correct][0]
                    weight_harder = WEIGHT_TABLE[q_diff][correct][1]
                    if q_diff == "easy":
                        questions_easier = concept_dict[c_id]["easy"][:k]
                        questions_harder = concept_dict[c_id]["hard"]
                    elif q_diff == "hard":
                        questions_easier = concept_dict[c_id]["easy"]
                        questions_harder = concept_dict[c_id]["hard"][k:]
                    else:
                        questions_easier = concept_dict[c_id]["easy"][:10]
                        questions_harder = concept_dict[c_id]["hard"][-10:]

                    q_easier = 0
                    c_easier = 0
                    q_harder = 0
                    c_harder = 0
                    if len(questions_easier) > 0:
                        q_easier = random.choice(questions_easier)
                        c_easier = c_id
                        mask_easier_seq.append(1)
                    else:
                        mask_easier_seq.append(0)

                    if len(questions_harder) > 0:
                        q_harder = random.choice(questions_harder)
                        c_harder = c_id
                        mask_harder_seq.append(1)
                    else:
                        mask_harder_seq.append(0)

                    question_easier_seq.append(q_easier)
                    question_harder_seq.append(q_harder)
                    concept_easier_seq.append(c_easier)
                    concept_harder_seq.append(c_harder)
                    weight_easier_seq.append(weight_easier)
                    weight_harder_seq.append(weight_harder)
                else:
                    raise NotImplementedError()

            question_easier_seq += [0] * pad_len
            question_harder_seq += [0] * pad_len
            weight_easier_seq += [0] * pad_len
            weight_harder_seq += [0] * pad_len
            mask_easier_seq += [0] * pad_len
            mask_harder_seq += [0] * pad_len

            result["question_easier_seq"] = torch.LongTensor(question_easier_seq).to(self.params["device"])
            result["question_harder_seq"] = torch.LongTensor(question_harder_seq).to(self.params["device"])
            result["weight_easier_seq"] = torch.FloatTensor(weight_easier_seq).to(self.params["device"])
            result["weight_harder_seq"] = torch.FloatTensor(weight_harder_seq).to(self.params["device"])
            result["mask_easier_seq"] = torch.LongTensor(mask_easier_seq).to(self.params["device"])
            result["mask_harder_seq"] = torch.LongTensor(mask_harder_seq).to(self.params["device"])
            result["correct_easier_seq"] = torch.ones(max_seq_len).long().to(self.params["device"])
            result["correct_harder_seq"] = torch.zeros(max_seq_len).long().to(self.params["device"])

            if data_type == "single_concept":
                concept_easier_seq += [0] * pad_len
                concept_harder_seq += [0] * pad_len
                result["concept_easier_seq"] = torch.LongTensor(concept_easier_seq).to(self.params["device"])
                result["concept_harder_seq"] = torch.LongTensor(concept_harder_seq).to(self.params["device"])
            else:
                raise NotImplementedError()

        return result

    def load_dataset(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        unuseful_keys = dataset_config_this["unuseful_seq_keys"]
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

        self.dataset = dataset_original
