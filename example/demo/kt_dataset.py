import torch.cuda
from torch.utils.data import DataLoader

from lib.util.FileManager import FileManager
from lib.dataset.KTDataset import KTDataset

if __name__ == "__main__":
    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "datasets_config": {
            # 当前dataset的选择
            "dataset_this": "train",
            "train": {
                # 两种数据格式，"kt" or "srs"，后者是序列推荐的格式
                "type": "kt",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_train_fold_0.txt",
                "dataset_path": "",
                "kt": {
                    # 配置KTDataset需要的参数
                    "data_type": "multi_concept",
                    "unuseful_seq_keys": {"user_id"},
                    "base_type": "concept"
                },
            },
            "valid": {
                "type": "kt",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_valid_fold_0.txt",
                "kt": {
                    "data_type": "multi_concept",
                    "unuseful_seq_keys": {"user_id"},
                    "base_type": "concept"
                },
            },
            "test": {
                "type": "kt",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_test_fold_0.txt",
                "kt": {
                    "data_type": "multi_concept",
                    "unuseful_seq_keys": {"user_id"},
                    "base_type": "concept"
                },
            }
        }
    }
    objects = {
        "file_manager": FileManager(r"F:\code\myProjects\dlkt")
    }

    dataset = KTDataset(params, objects)
    dataloader = DataLoader(dataset, batch_size=32)

    for batch in dataloader:
        print("")
