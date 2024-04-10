import os


def drop_prefix(target_dir, prefix2drop):
    all_files_in_dir = os.listdir(target_dir)
    for file_name in all_files_in_dir:
        if file_name.startswith(prefix2drop):
            old_path = os.path.join(target_dir, file_name)
            new_path = os.path.join(target_dir, file_name.replace(prefix2drop, ""))
            os.rename(old_path, new_path)
            continue


if __name__ == "__main__":
    drop_prefix(r"F:\code\myProjects\dlkt\example\result_local\our_setting_new\random-predictor", "random-predictor_our_setting_new_")
