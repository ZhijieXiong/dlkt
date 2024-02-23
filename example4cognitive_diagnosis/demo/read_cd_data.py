import torch.cuda
from torch.utils.data import DataLoader

from lib.util.FileManager import FileManager
from lib.dataset.CDDataset import CDDataset


if __name__ == "__main__":
    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "datasets_config": {
            # 当前dataset的选择
            "dataset_this": "train",
            "train": {
                "type": "cd",
                "setting_name": "ncd_setting",
                "file_name": "assist2009_train_fold_0.txt",
            },
            "valid": {
                "type": "cd",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_valid_fold_0.txt",
            },
            "test": {
                "type": "cd",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_test_fold_0.txt",
            }
        }
    }
    objects = {
        "file_manager": FileManager(r"F:\code\myProjects\dlkt")
    }

    dataset = CDDataset(params, objects)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    for batch in dataloader:
        print("")
