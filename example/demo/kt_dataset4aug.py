import torch.cuda
from torch.utils.data import DataLoader

from lib.util.FileManager import FileManager
from lib.dataset.KTDataset4Aug import KTDataset4Aug


if __name__ == "__main__":
    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "datasets_config": {
            # 当前dataset的选择
            "dataset_this": "train",
            "data_type": "multi_concept",
            "train": {
                # 两种数据格式，"kt" or "srs"，后者是序列推荐的格式
                "type": "kt4aug",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_train_fold_0.txt",
                "unuseful_seq_keys": {"user_id"},
                "kt": {
                    # 配置KTDataset需要的参数
                    "base_type": "concept"
                },
                "kt4aug": {
                    # "random_aug" or "semantic_aug" or "informative_aug"
                    "aug_type": "informative_aug",
                    "num_aug": 2,
                    "random_aug": {
                        # 配置随机增强
                        "mask_prob": 0.1,
                        "replace_prob": 0.1,
                        "crop_prob": 0.1,
                        "permute_prob": 0.1,
                        "hard_neg_prob": 1.0,
                        "aug_order": ["mask", "replace", "permute", "crop"]
                    },
                    "informative_aug": {
                        # 配置info增强
                        "mask_prob": 0.1,
                        "replace_prob": 0.1,
                        "crop_prob": 0.1,
                        "insert_prob": 0.1,
                        # "order" or "transmission"
                        "offline_sim_type": "order",
                        "num_concept": 123,
                        "num_question": 17751,
                        # "offline" or "online" or "hybrid"
                        "sim_type": "off",
                        "aug_order": ["mask", "crop", "replace", "insert"]
                    }
                }
            },
            "valid": {
                "type": "kt",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_valid_fold_0.txt",
                "unuseful_seq_keys": {"user_id"},
                "kt": {
                    "base_type": "concept"
                },
            },
            "test": {
                "type": "kt",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_test_fold_0.txt",
                "unuseful_seq_keys": {"user_id"},
                "kt": {
                    "base_type": "concept"
                },
            }
        },
        "other": {
            "duo": {
              "temp": 0.05
            },
            "informative_aug_config": {
              "num_concept": 123,
              "num_question": 17751,
            }
        }
    }
    objects = {
        "file_manager": FileManager(r"F:\code\myProjects\dlkt"),
        "data": {
            "Q_table": FileManager(r"F:\code\myProjects\dlkt").get_q_table("assist2009", "multi_concept")
        }
    }

    dataset = KTDataset4Aug(params, objects)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        print("")