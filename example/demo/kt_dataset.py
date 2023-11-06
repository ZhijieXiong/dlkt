import torch.cuda
from torch.utils.data import DataLoader

from lib.dataset.KTDataset import KTDataset


if __name__ == "__main__":
    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset_this_config": {
            "dataset_path": r"F:\code\myProjects\dlkt\lab\settings\cl4kt_setting\assist2009_train_fold_0.txt",
            "data_type": "single_concept",
            "unuseful_seq_keys": {"user_id", "school_id", "teacher_id", "seq_len"},
            "base_type": "concept"
        }
    }
    objects = {}

    dataset = KTDataset(params, objects)
    dataloader = DataLoader(dataset, batch_size=32)

    for batch in dataloader:
        print("")
