import torch
import random
from torch.utils.data import Dataset
from collections import Counter

from ..util.data import *
from ..util.parse import *
from .util import data_kt2srs


class KTDataset4MELT(Dataset):
    def __init__(self, params, objects):
        super(KTDataset4MELT, self).__init__()
        self.params = params
        self.objects = objects

        self.data_uniformed = None
        self.dataset = None
        self.data_srs = None
        self.head_questions = None
        self.load_dataset()

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

    def parse_long_tail(self):
        """
        解析数据集的长尾
        :return:
        """
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        high_threshold = dataset_config_this["long_tail"]["high_threshold"]
        data_type = self.params["datasets_config"]["data_type"]
        data_srs = data_kt2srs(self.data_uniformed, data_type)
        self.data_srs = data_srs

        question_seqs = list(map(lambda item_data: item_data["question_seq"][:item_data["seq_len"]], self.data_uniformed))
        questions = []
        for question_seq in question_seqs:
            questions += question_seq
        question_frequency = Counter(questions)
        d_list = list(question_frequency.items())
        d_list = sorted(d_list, key=lambda x: x[1])
        d_list = list(map(lambda x: x[0], d_list))
        num_question = len(d_list)
        self.head_questions = d_list[int(num_question * high_threshold):]




