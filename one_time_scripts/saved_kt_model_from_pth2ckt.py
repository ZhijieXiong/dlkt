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


def rm_pth(path_or_dir):
    if os.path.isfile(path_or_dir):
        file_name = os.path.basename(path_or_dir)
        if file_name == "kt_model.pth":
            try:
                os.remove(path_or_dir)
            except:
                print(f"error: {path_or_dir}")
    else:
        for f in os.listdir(path_or_dir):
            rm_pth(os.path.join(path_or_dir, f))


if __name__ == "__main__":
    root_dir = r"F:\code\myProjects\dlkt\lab\saved_models"
    # convert_pth2ckt(root_dir)
    # 运行结果
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@01-26-23@@DKVMN@@seed_0@@our_setting@@assist2012_train_fold_0\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@01-43-10@@DKVMN@@seed_0@@our_setting@@assist2012_train_fold_1\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@01-53-36@@DKVMN@@seed_0@@our_setting@@assist2012_train_fold_2\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@02-06-03@@DKVMN@@seed_0@@our_setting@@assist2012_train_fold_3\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@02-16-43@@DKVMN@@seed_0@@our_setting@@assist2012_train_fold_4\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@02-29-11@@DKVMN@@seed_0@@our_setting@@assist2017_train_fold_0\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@02-35-10@@DKVMN@@seed_0@@our_setting@@assist2017_train_fold_1\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@02-39-26@@DKVMN@@seed_0@@our_setting@@assist2017_train_fold_2\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@02-44-48@@DKVMN@@seed_0@@our_setting@@assist2017_train_fold_3\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@02-50-59@@DKVMN@@seed_0@@our_setting@@assist2017_train_fold_4\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@02-55-42@@DKVMN@@seed_0@@our_setting@@edi2020-task34_train_fold_0\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-00-20@@DKVMN@@seed_0@@our_setting@@edi2020-task34_train_fold_1\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-05-20@@DKVMN@@seed_0@@our_setting@@edi2020-task34_train_fold_2\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-09-42@@DKVMN@@seed_0@@our_setting@@edi2020-task34_train_fold_3\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-14-04@@DKVMN@@seed_0@@our_setting@@edi2020-task34_train_fold_4\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-19-17@@DKVMN@@seed_0@@our_setting@@slepemapy_train_fold_0\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-25-08@@DKVMN@@seed_0@@our_setting@@slepemapy_train_fold_1\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-31-55@@DKVMN@@seed_0@@our_setting@@slepemapy_train_fold_2\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-37-03@@DKVMN@@seed_0@@our_setting@@slepemapy_train_fold_3\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-41-38@@DKVMN@@seed_0@@our_setting@@slepemapy_train_fold_4\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-46-40@@DKVMN@@seed_0@@our_setting@@statics2011_train_fold_0\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-47-29@@DKVMN@@seed_0@@our_setting@@statics2011_train_fold_1\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-48-10@@DKVMN@@seed_0@@our_setting@@statics2011_train_fold_2\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-48-50@@DKVMN@@seed_0@@our_setting@@statics2011_train_fold_3\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\DKVMN\2024-02-17@03-49-34@@DKVMN@@seed_0@@our_setting@@statics2011_train_fold_4\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@21-33-57@@LPKT+@@seed_0@@our_setting@@statics2011_train_fold_0\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@21-36-42@@LPKT+@@seed_0@@our_setting@@statics2011_train_fold_1\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@21-39-33@@LPKT+@@seed_0@@our_setting@@statics2011_train_fold_2\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@21-42-24@@LPKT+@@seed_0@@our_setting@@statics2011_train_fold_3\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@21-45-22@@LPKT+@@seed_0@@our_setting@@statics2011_train_fold_4\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@21-48-19@@LPKT+@@seed_0@@our_setting@@ednet-kt1_train_fold_0\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@21-54-43@@LPKT+@@seed_0@@our_setting@@ednet-kt1_train_fold_1\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@22-00-47@@LPKT+@@seed_0@@our_setting@@ednet-kt1_train_fold_2\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@22-07-11@@LPKT+@@seed_0@@our_setting@@ednet-kt1_train_fold_3\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@22-12-53@@LPKT+@@seed_0@@our_setting@@ednet-kt1_train_fold_4\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@22-17-23@@LPKT+@@seed_0@@our_setting@@assist2017_train_fold_0\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@22-33-49@@LPKT+@@seed_0@@our_setting@@assist2017_train_fold_1\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@22-52-06@@LPKT+@@seed_0@@our_setting@@assist2017_train_fold_2\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@23-05-47@@LPKT+@@seed_0@@our_setting@@assist2017_train_fold_3\kt_model.pth
    # error: F:\code\myProjects\dlkt\lab\saved_models\save\our_setting\LPKT+_IRT_not_share\2024-03-06@23-20-46@@LPKT+@@seed_0@@our_setting@@assist2017_train_fold_4\kt_model.pth
    # rm_pth(root_dir)
