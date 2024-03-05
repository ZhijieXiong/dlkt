from torch.utils.data import Dataset

from ..util.data import *
from ..util.parse import *
from .sample_weight import *
from ..CONSTANT import INTERVAL_TIME4LPKT_PLUS, USE_TIME4LPKT_PLUS


class KTDataset4LPKTPlus(Dataset):
    def __init__(self, params, objects):
        super(KTDataset4LPKTPlus, self).__init__()
        self.params = params
        self.objects = objects
        self.dataset = None
        self.label4lpkt_plus = {}

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
        if "time_seq" in seq_keys:
            dataset_converted["interval_time_seq"] = []
        dataset_converted["seq_id"] = []
        if self.params.get("use_sample_weight", False):
            dataset_converted["weight_seq"] = []
        max_seq_len = len(dataset_original[0]["mask_seq"])
        for seq_i, item_data in enumerate(dataset_original):
            for k in id_keys:
                dataset_converted[k].append(item_data[k])
            for k in seq_keys:
                if k == "time_seq":
                    interval_time_seq = [0]
                    seq_len = item_data["seq_len"]
                    for time_i in range(1, seq_len):
                        interval_time_real = (item_data["time_seq"][time_i] - item_data["time_seq"][time_i - 1]) // 60
                        interval_time_idx = len(INTERVAL_TIME4LPKT_PLUS)
                        for idx, interval_time_value in enumerate(INTERVAL_TIME4LPKT_PLUS):
                            if interval_time_real < 0:
                                interval_time_idx = 0
                                break
                            if interval_time_real <= interval_time_value:
                                interval_time_idx = idx
                                break
                        interval_time_seq.append(interval_time_idx)
                    interval_time_seq += [0] * (max_seq_len - seq_len)
                    dataset_converted["interval_time_seq"].append(interval_time_seq)
                elif k == "use_time_seq":
                    seq_len = item_data["seq_len"]
                    use_time_seq = []
                    for time_i, use_time in enumerate(item_data["use_time_seq"][:seq_len]):
                        use_time_idx = len(USE_TIME4LPKT_PLUS)
                        for idx, use_time_value in enumerate(USE_TIME4LPKT_PLUS):
                            if use_time <= use_time_value:
                                use_time_idx = idx
                                break
                        use_time_seq.append(use_time_idx)
                    use_time_seq += [0] * (max_seq_len - seq_len)
                    dataset_converted["use_time_seq"].append(use_time_seq)
                else:
                    dataset_converted[k].append(item_data[k])

            # 生成weight seq
            if self.params.get("use_sample_weight", False):
                if self.params["sample_weight_method"] == "discount":
                    w_seq = discount(item_data["correct_seq"], item_data["seq_len"], max_seq_len)
                elif self.params["sample_weight_method"] == "highlight_tail":
                    w_seq = highlight_tail(self.params["tail_weight"], item_data["question_seq"],
                                           self.objects["data"]["train_data_statics"], item_data["seq_len"],
                                           max_seq_len)
                else:
                    raise NotImplementedError()
                dataset_converted["weight_seq"].append(w_seq)
            dataset_converted["seq_id"].append(seq_i)

        if "time_seq" in dataset_converted.keys():
            del dataset_converted["time_seq"]

        for k in dataset_converted.keys():
            if k not in ["weight_seq"]:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.params["device"])
            else:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).float().to(self.params["device"])
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

    @staticmethod
    def parse_difficulty(data_uniformed, data_type, qc_difficulty):
        if "question_diff_seq" in data_uniformed[0].keys():
            # 如果是用的dimkt_setting处理的数据，习题和知识点难度已经预处理好了
            return
        # 目前只考虑single concept数据集
        question_difficulty, concept_difficulty = qc_difficulty
        for item_data in data_uniformed:
            item_data["question_diff_seq"] = []
            for q_id in item_data["question_seq"]:
                item_data["question_diff_seq"].append(question_difficulty[q_id])

            if data_type != "only_question":
                item_data["concept_diff_seq"] = []
                for c_id in item_data["concept_seq"]:
                    item_data["concept_diff_seq"].append(concept_difficulty[c_id])
