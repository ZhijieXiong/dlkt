import argparse
import torch

from torch.utils.data import DataLoader

from evaluate_config import evaluate_general_config
from kg4ex_util import load_dkt

from lib.dataset.KTDataset import KTDataset
from lib.util.parse import str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device配置
    parser.add_argument("--use_cpu", type=str2bool, default=False)

    # 加载模型参数配置
    parser.add_argument("--save_model_dir", type=str, help="绝对路径",
                        default=r"F:\code\myProjects\dlkt\lab\saved_models\kg4ex\2024-07-30@14-54-54@@DKT@@seed_0@@our_setting_new@@statics2011_train_fold_0")
    parser.add_argument("--save_model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")

    # 测试配置
    parser.add_argument("--setting_name", type=str, default="our_setting_new")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--test_file_name", type=str, default=r"statics2011_test_fold_0.txt")
    parser.add_argument("--base_type", type=str, default="concept")
    parser.add_argument("--evaluate_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=200)

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = evaluate_general_config(params)

    dataset = KTDataset(global_params, global_objects)
    data_loader = DataLoader(dataset, batch_size=params["evaluate_batch_size"], shuffle=False)

    model = load_dkt(global_params, global_objects, params["save_model_dir"], params["save_model_name"], params["model_name_in_ckt"])
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            user_mlkc = model.get_last_mlkc_kg4ex(batch)
