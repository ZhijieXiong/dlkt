import os
import shutil


def rename_file(setting_dir):
    # [dataset_name]_[multi | single]_train.txt --> [dataset_name]_[multi | single]_train_fold_0.txt
    # [dataset_name]_[multi | single]_test_iid.txt --> [dataset_name]_[multi | single]_valid_fold_0.txt
    # [dataset_name]_[multi | single]_test_ood.txt --> [dataset_name]_[multi | single]_test.txt
    all_files_in_setting = os.listdir(setting_dir)
    for file_name in all_files_in_setting:
        if "test_iid.txt" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            new_name = os.path.join(setting_dir, file_name.replace("test_iid", "valid_fold_0"))
            os.rename(old_name, new_name)
            continue
        if "test_ood.txt" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            new_name = os.path.join(setting_dir, file_name.replace("_ood", ""))
            os.rename(old_name, new_name)
            continue
        if "train.txt" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            new_name = os.path.join(setting_dir, file_name.replace("train", "train_fold_0"))
            os.rename(old_name, new_name)
            continue


def copy_file_all(src_dir, dst_dir):
    all_files_in_setting = os.listdir(src_dir)
    for file_name in all_files_in_setting:
        src_file_path = os.path.join(src_dir, file_name)
        dst_file_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_file_path, dst_file_path)
