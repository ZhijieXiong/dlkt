import random

from torch.utils.data import Dataset

from ..util.data import *
from ..util.parse import *

# S -> predict score
# S_easy - S_easier >= 0, weight: 0.1
# S_easy - S_hard <= 0, weight: 0.5
# S_hard - S_easy >= 0, weight: 0.5
# S_hard - S_harder <= 0, weight: 0.1
# S_middle - S_easy >= 0, weight: 0.2
# S_middle - S_hard <= 0, weight: 0.2
WEIGHT_TABLE = {
    "unknown": (0.3, 0.3),
    "easy": (0.1, 1),
    "hard": (1, 0.1),
    "middle": (0.5, 0.5),
}
MASK_TABLE = {k: tuple(map(lambda x: 1 if x != 0 else 0, WEIGHT_TABLE[k])) for k, v in WEIGHT_TABLE.items()}


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

        if kt_dataset_type == "kt_output_enhance":
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
                if data_type == "single_concept":
                    c_pair = question_dict[q_id][1][0]
                    c_id = c_pair[0]
                    q_diff = c_pair[1]
                    k = c_pair[2]
                    weight_easier = WEIGHT_TABLE[q_diff][0]
                    weight_harder = WEIGHT_TABLE[q_diff][1]
                    mask_easier = MASK_TABLE[q_diff][0]
                    mask_harder = MASK_TABLE[q_diff][1]
                    if q_diff == "easy":
                        questions_easier = concept_dict[c_id]["easy"][:k]
                        questions_harder = concept_dict[c_id]["hard"]
                    elif q_diff == "hard":
                        questions_easier = concept_dict[c_id]["easy"]
                        questions_harder = concept_dict[c_id]["hard"][k:]
                    elif q_diff == "middle":
                        questions_easier = concept_dict[c_id]["easy"][:10]
                        questions_harder = concept_dict[c_id]["hard"][-10:]
                    else:
                        questions_easier = concept_dict[c_id]["easy"][:3]
                        questions_harder = concept_dict[c_id]["hard"][-3:]

                    q_easier = 0
                    c_easier = 0
                    q_harder = 0
                    c_harder = 0
                    if len(questions_easier) > 0:
                        q_easier = random.choice(questions_easier)
                        c_easier = c_id
                    if len(questions_harder) > 0:
                        q_harder = random.choice(questions_harder)
                        c_harder = c_id

                    question_easier_seq.append(q_easier)
                    question_harder_seq.append(q_harder)
                    concept_easier_seq.append(c_easier)
                    concept_harder_seq.append(c_harder)
                    weight_easier_seq.append(weight_easier)
                    weight_harder_seq.append(weight_harder)
                    mask_easier_seq.append(mask_easier)
                    mask_harder_seq.append(mask_harder)
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

    def get_statics_kt_dataset(self):
        num_seq = len(self.dataset)
        num_sample = 0
        num_correct = 0
        for item_data in self.dataset:
            num_sample += item_data["seq_len"]
            num_correct += sum(item_data["correct_seq"])
        return num_seq, num_sample, num_correct / num_sample
