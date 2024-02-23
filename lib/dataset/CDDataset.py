import os
import torch

from torch.utils.data import Dataset

from ..util.data import read_cd_task_dataset


class CDDataset(Dataset):
    def __init__(self, params, objects):
        super(CDDataset, self).__init__()
        self.params = params
        self.objects = objects
        self.dataset = None
        self.load_dataset()

    def __len__(self):
        return len(self.dataset["user_id"])

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

        if dataset_path != "":
            dataset_original = read_cd_task_dataset(dataset_path)
        else:
            dataset_original = self.objects["dataset_this"]

        all_keys = list(dataset_original[0].keys())
        dataset_converted = {k: [] for k in all_keys}
        for interaction_data in dataset_original:
            for k in all_keys:
                dataset_converted[k].append(interaction_data[k])

        for k in dataset_converted.keys():
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
