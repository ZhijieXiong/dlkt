from torch.utils.data import Dataset

from ..util.data import *
from ..util.parse import *
from .util import data_kt2srs
from .KTDataset import KTDataset


class SRSDataset4KT(Dataset):
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.data_uniformed = None
        self.dataset = None

        self.load_dataset()

    def __getitem__(self, idx):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]["srs"]
        max_seq_len = dataset_config_this["max_seq_len"]

        item_meta = self.dataset[idx]
        seq_id = item_meta["target_seq_id"]
        seq_len = item_meta["target_seq_len"]
        item_data = self.data_uniformed[seq_id]
        result = {
            "seq_id": torch.tensor(seq_id).long().to(self.params["device"]),
            "seq_len": torch.tensor(seq_len).long().to(self.params["device"]),
        }

        for k in item_data.keys():
            if k in ["question_seq", "concept_seq", "correct_seq", "use_time_seq", "interval_time_seq"]:
                result[k] = torch.tensor(
                    item_data[k][:seq_len] + [0] * (max_seq_len - seq_len)
                ).long().to(self.params["device"])

        for k in item_meta.keys():
            if k in ["target_concept", "target_question", "target_correct", "target_time", "target_interval_time",
                     "target_use_time", "target_question_diff", "target_concept_diff"]:
                result[k] = torch.tensor(item_meta[k]).long().to(self.params["device"])

        return result

    def __len__(self):
        return len(self.dataset)

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

        use_diff4dimkt = dataset_config_this["srs"].get("use_diff4dimkt", False)
        if use_diff4dimkt:
            question_difficulty = self.objects["dimkt"]["question_difficulty"]
            concept_difficulty = self.objects["dimkt"]["concept_difficulty"]
            qc_difficulty = (question_difficulty, concept_difficulty)
            KTDataset.parse_difficulty(dataset_original, data_type, qc_difficulty)

        if data_type == "multi_concept":
            self.data_uniformed = data_agg_question(dataset_original)
        else:
            self.data_uniformed = deepcopy(dataset_original)

        id_keys, seq_keys = get_keys_from_uniform(self.data_uniformed)
        all_keys = set(id_keys).union(seq_keys)
        id_keys = list(set(id_keys) - unuseful_keys)
        seq_keys = list(set(seq_keys) - unuseful_keys - {"age_seq"})
        unuseful_keys = all_keys - set(id_keys).union(set(seq_keys))
        for item_data in self.data_uniformed:
            for k in unuseful_keys:
                del item_data[k]

        if "time_seq" in seq_keys:
            for item_data in self.data_uniformed:
                interval_time_seq = [0]
                for time_i in range(1, len(item_data["time_seq"])):
                    interval_time = (item_data["time_seq"][time_i] - item_data["time_seq"][time_i - 1]) // 60
                    interval_time = max(0, min(interval_time, 60 * 24 * 30))
                    interval_time_seq.append(interval_time)
                item_data["interval_time_seq"] = interval_time_seq

        self.dataset = data_kt2srs(self.data_uniformed)

    def get_statics_kt_dataset(self):
        num_seq = len(self.data_uniformed)
        num_sample = len(self.dataset) + num_seq
        num_right = 0
        for item_data in self.data_uniformed:
            num_right += sum(item_data["correct_seq"][:item_data["seq_len"]])

        return num_seq, num_sample, num_right / num_sample
