import os
import json


from .util.data import load_json


class FileManager:
    dataset_raw_path_in_lab = {
        "assist2009": "lab/dataset_raw/assist2009/skill_builder_data_corrected_collapsed.csv",
        "assist2009-new": "lab/dataset_raw/assist2009-new/skill_builder_data.csv",
        "assist2012": "lab/dataset_raw/assist2012/2012-2013-data-with-predictions-4-final.csv",
        "assist2015": "lab/dataset_raw/assist2015/2015_100_skill_builders_main_problems.csv",
        "assist2017": "lab/dataset_raw/assist2017/anonymized_full_release_competition_dataset.csv",
        "edi2020-task1": "lab/dataset_raw/edi2020",
        "edi2020-task12": "lab/dataset_raw/edi2020",
        "edi2020-task4": "lab/dataset_raw/edi2020",
        "edi2022": "lab/dataset_raw/edi2022",
        "SLP-bio": "lab/dataset_raw/SLP",
        "SLP-chi": "lab/dataset_raw/SLP",
        "SLP-eng": "lab/dataset_raw/SLP",
        "SLP-mat": "lab/dataset_raw/SLP",
        "SLP-his": "lab/dataset_raw/SLP",
        "SLP-geo": "lab/dataset_raw/SLP",
        "SLP-phy": "lab/dataset_raw/SLP",
        "slepemapy": "lab/dataset_raw/slepemapy/answers.csv",
        "statics2011": "lab/dataset_raw/statics2011/AllData_student_step_2011F.csv",
        "ednet-kt1": "lab/dataset_raw/ednet-kt1"
    }

    data_preprocessed_dir_in_lab = {
        "assist2009": "lab/dataset_preprocessed/assist2009",
        "assist2009-new": "lab/dataset_preprocessed/assist2009-new",
        "assist2012": "lab/dataset_preprocessed/assist2012",
        "assist2015": "lab/dataset_preprocessed/assist2015",
        "assist2017": "lab/dataset_preprocessed/assist2017",
        "edi2020-task1": "lab/dataset_preprocessed/edi2020-task1",
        "edi2020-task12": "lab/dataset_preprocessed/edi2020-task12",
        "edi2020-task4": "lab/dataset_preprocessed/edi2020-task4",
        "edi2022": "lab/dataset_preprocessed/edi2020",
        "SLP-bio": "lab/dataset_preprocessed/SLP-bio",
        "SLP-chi": "lab/dataset_preprocessed/SLP-chi",
        "SLP-eng": "lab/dataset_preprocessed/SLP-eng",
        "SLP-mat": "lab/dataset_preprocessed/SLP-mat",
        "SLP-his": "lab/dataset_preprocessed/SLP-his",
        "SLP-geo": "lab/dataset_preprocessed/SLP-geo",
        "SLP-phy": "lab/dataset_preprocessed/SLP-phy",
        "statics2011": "lab/dataset_preprocessed/statics2011",
        "ednet-kt1": "lab/dataset_preprocessed/ednet-kt1",
        "junyi2015": "lab/dataset_preprocessed/junyi2015",
        "slepemapy": "lab/dataset_preprocessed/slepemapy"
    }

    data_preprocessed_name = {
        "multi": "data_multi.txt",
        "single": "data_single.txt",
    }

    builtin_datasets = ["assist2009", "assist2009-new", "assist2012", "assist2015", "assist2017", "statics2011",
                        "junyi2015", "ednet-kt1", "edi2020", "edi2020-task1", "edi2020-task12", "edi2020-task4",
                        "edi2022", "SLP-bio", "SLP-mat", "slepemapy"]

    result_dir_in_lab = "lab/result"
    models_dir_in_lab = "lab/result/saved_models"
    file_settings_name = "setting.json"
    data_info_name = "info.json"
    Q_table_name = "Q_table.npy"

    def __init__(self, root_dir, init_dirs=False):
        self.root_dir = root_dir
        if init_dirs:
            self.create_dirs()

    def create_dirs(self):
        assert os.path.exists(self.root_dir), f"{self.root_dir} not exist"
        assert not os.path.exists(os.path.join(self.root_dir, "lab")), f"lab already exists in {self.root_dir}"
        all_dirs = [
            os.path.join(self.root_dir, "lab"),
            os.path.join(self.root_dir, "lab", "dataset_raw"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "assist2009"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "assist2009-new"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "assist2012"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "assist2015"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "assist2017"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "edi2020"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "edi2022"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "statics2011"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "junyi2015"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "ednet-kt1"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "kdd_cup2010"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "SLP"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "slepemapy"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2009"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2009-new"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2012"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2015"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2017"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "statics2011"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "edi2020-task1"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "edi2020-task12"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "edi2020-task4"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "junyi2015"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "ednet-kt1"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "kdd_cup2010"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "SLP-bio"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "SLP-chi"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "SLP-eng"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "SLP-his"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "SLP-mat"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "SLP-geo"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "SLP-phy"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "slepemapy"),
            os.path.join(self.root_dir, "lab", "result"),
            os.path.join(self.root_dir, "lab", "result", "saved_models"),
        ]
        # win系统下是用'\\'分隔的
        all_dirs = list(sorted(all_dirs, key=lambda dir_str: len(dir_str.split("\\"))))
        for dir_ in all_dirs:
            if os.path.join(dir_):
                os.mkdir(dir_)

    def get_root_dir(self):
        return self.root_dir

    def get_dataset_raw_path(self, dataset_name):
        return os.path.join(self.root_dir, FileManager.dataset_raw_path_in_lab[dataset_name])

    # ==================================================================================================================
    def get_preprocessed_dir(self, dataset_name):
        assert dataset_name in FileManager.builtin_datasets, f"{dataset_name} is not in builtin datasets"
        return os.path.join(self.root_dir, FileManager.data_preprocessed_dir_in_lab[dataset_name])

    def get_data_info_path(self, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        return os.path.join(preprocessed_dir, FileManager.data_info_name)

    def get_q_table_path(self, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        return os.path.join(preprocessed_dir, FileManager.Q_table_name)

    def get_data_preprocessed_info(self, dataset_name):
        data_info_path = self.get_data_info_path(dataset_name)
        info = load_json(data_info_path)
        return info["preprocessed"]

    def get_preprocessed_path(self, dataset_name, multi=True):
        # 根据数据集选择
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        file_name_key = 'multi' if multi else 'single'
        return os.path.join(preprocessed_dir, FileManager.data_preprocessed_name[file_name_key])

    # ==================================================================================================================

    def get_models_dir(self):
        return os.path.join(self.root_dir, FileManager.models_dir_in_lab)

    # ==================================================================================================================
    def add_new_setting(self, setting_new, setting_new_name):
        setting_dir = os.path.join(self.root_dir, FileManager.result_dir_in_lab, setting_new_name)
        # assert not os.path.exists(setting_dir) and not os.path.isdir(setting_dir), f"{setting_new_name} dir exists"
        if os.path.exists(setting_dir) and os.path.isdir(setting_dir):
            return
        setting_path = os.path.join(setting_dir, FileManager.file_settings_name)
        os.mkdir(setting_dir)
        with open(setting_path, "w") as f:
            json.dump(setting_new, f, indent=2)

    def delete_old_setting(self, setting_old_name):
        setting_dir = os.path.join(self.root_dir, FileManager.result_dir_in_lab, setting_old_name)
        assert os.path.exists(setting_dir) or os.path.isdir(setting_dir), f"{setting_old_name} dir does not exist"
        os.rmdir(setting_dir)

    def get_setting_dir(self, setting_name):
        result_dir = os.path.join(self.root_dir, FileManager.result_dir_in_lab)
        setting_names = os.listdir(result_dir)
        assert setting_name in setting_names, f"{setting_name} dir does not exist"
        return os.path.join(result_dir, setting_name)

    def get_setting_file_path(self, setting_name):
        setting_dir = self.get_setting_dir(setting_name)
        return os.path.join(setting_dir, FileManager.file_settings_name)
    # ==================================================================================================================
