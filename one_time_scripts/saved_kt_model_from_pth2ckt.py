"""
之前存储模型用的是torch.save(model_instance, path)，直接将整个object存储
后面改成存储model state_dict，所以需要加载已经训练好的模型并保存为ckt
该脚本的作用是将一个目录下所有（递归）的pth文件加载，然后保存为ckt
"""

import os
import torch


def convert_pth2ckt(path_or_dir):
    if os.path.isfile(path_or_dir):
        file_name = os.path.basename(path_or_dir)
        file_dir = os.path.dirname(path_or_dir)
        if file_name == "kt_model.pth":
            try:
                model = torch.load(path_or_dir)
                model_weight_path = os.path.join(file_dir, "saved.ckt")
                torch.save({"best_valid": model.state_dict()}, model_weight_path)
            except:
                print(f"error: {path_or_dir}")
    else:
        for f in os.listdir(path_or_dir):
            convert_pth2ckt(os.path.join(path_or_dir, f))


if __name__ == "__main__":
    root_dir = r"F:\code\myProjects\dlkt\lab\saved_models"
    convert_pth2ckt(root_dir)
