import os
import json
import platform
import numpy as np


from .data import load_json, write_json
from ..data_processor.load_raw import load_csv


class FileManager:
    dataset_raw_path_in_lab = {
        "assist2009": "lab/dataset_raw/assist2009/skill_builder_data_corrected_collapsed.csv",
        "assist2009-new": "lab/dataset_raw/assist2009-new/skill_builder_data_corrected_collapsed.csv",
        "assist2012": "lab/dataset_raw/assist2012/2012-2013-data-with-predictions-4-final.csv",
        "assist2015": "lab/dataset_raw/assist2015/2015_100_skill_builders_main_problems.csv",
        "assist2017": "lab/dataset_raw/assist2017/anonymized_full_release_competition_dataset.csv",
        "edi2020-task1": "lab/dataset_raw/edi2020",
        "edi2020-task2": "lab/dataset_raw/edi2020",
        "edi2020-task34": "lab/dataset_raw/edi2020",
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
        "ednet-kt1": "lab/dataset_raw/ednet-kt1",
        "xes3g5m": "lab/dataset_raw/xes3g5m",
        "aaai2023": "lab/dataset_raw/aaai2023",
        "algebra2005": "lab/dataset_raw/kdd_cup2010",
        "algebra2006": "lab/dataset_raw/kdd_cup2010",
        "algebra2008": "lab/dataset_raw/kdd_cup2010",
        "bridge2algebra2006": "lab/dataset_raw/kdd_cup2010",
        "bridge2algebra2008": "lab/dataset_raw/kdd_cup2010"
    }

    data_preprocessed_dir_in_lab = {
        "assist2009": "lab/dataset_preprocessed/assist2009",
        "assist2009-new": "lab/dataset_preprocessed/assist2009-new",
        "assist2012": "lab/dataset_preprocessed/assist2012",
        "assist2015": "lab/dataset_preprocessed/assist2015",
        "assist2017": "lab/dataset_preprocessed/assist2017",
        "edi2020-task1": "lab/dataset_preprocessed/edi2020-task1",
        "edi2020-task2": "lab/dataset_preprocessed/edi2020-task12",
        "edi2020-task34": "lab/dataset_preprocessed/edi2020-task34",
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
        "slepemapy": "lab/dataset_preprocessed/slepemapy",
        "xes3g5m": "lab/dataset_preprocessed/xes3g5m",
        "aaai2023": "lab/dataset_preprocessed/aaai2023",
        "algebra2005": "lab/dataset_preprocessed/algebra2005",
        "algebra2006": "lab/dataset_preprocessed/algebra2006",
        "algebra2008": "lab/dataset_preprocessed/algebra2008",
        "bridge2algebra2006": "lab/dataset_preprocessed/bridge2algebra2006",
        "bridge2algebra2008": "lab/dataset_preprocessed/bridge2algebra2008"
    }

    builtin_datasets = ["assist2009", "assist2009-new", "assist2012", "assist2015", "assist2017", "statics2011",
                        "junyi2015", "ednet-kt1", "edi2020", "edi2020-task1", "edi2020-task2", "edi2020-task34",
                        "edi2022", "SLP-bio", "SLP-mat", "slepemapy", "xes3g5m", "algebra2005", "bridge2algebra2006"]

    setting_dir_in_lab = "lab/settings"
    models_dir_in_lab = "lab/saved_models"
    file_settings_name = "setting.json"

    def __init__(self, root_dir, init_dirs=False):
        self.root_dir = root_dir
        if init_dirs:
            self.create_dirs()

    def create_dirs(self):
        assert os.path.exists(self.root_dir), f"{self.root_dir} not exist"
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
            os.path.join(self.root_dir, "lab", "dataset_raw", "xes3g5m"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "aaai2023"),
            os.path.join(self.root_dir, "lab", "dataset_raw", "kdd_cup2010"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2009"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2009-new"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2012"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2015"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "assist2017"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "statics2011"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "edi2020-task1"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "edi2020-task2"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "edi2020-task34"),
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
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "xes3g5m"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "aaai2023"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "algebra2005"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "algebra2006"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "algebra2008"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "bridge2algebra2006"),
            os.path.join(self.root_dir, "lab", "dataset_preprocessed", "bridge2algebra2008"),
            os.path.join(self.root_dir, "lab", "settings"),
            os.path.join(self.root_dir, "lab", "saved_models"),
        ]
        if platform.system() == "Windows":
            all_dirs = list(sorted(all_dirs, key=lambda dir_str: len(dir_str.split("\\"))))
        else:
            all_dirs = list(sorted(all_dirs, key=lambda dir_str: len(dir_str.split("/"))))
        for dir_ in all_dirs:
            if not os.path.exists(dir_):
                os.mkdir(dir_)

    def get_root_dir(self):
        return self.root_dir

    def get_dataset_raw_path(self, dataset_name):
        return os.path.join(self.root_dir, FileManager.dataset_raw_path_in_lab[dataset_name])

    # ==================================================================================================================
    def get_preprocessed_dir(self, dataset_name):
        assert dataset_name in FileManager.builtin_datasets, f"{dataset_name} is not in builtin datasets"
        return os.path.join(self.root_dir, FileManager.data_preprocessed_dir_in_lab[dataset_name])

    def save_q_table(self, Q_table, dataset_name, data_type):
        assert data_type in ["multi_concept", "single_concept", "only_question"]
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        Q_table_path = os.path.join(preprocessed_dir, f"Q_table_{data_type}.npy")
        np.save(Q_table_path, Q_table)

    def save_data_statics_processed(self, statics, dataset_name, data_type):
        assert data_type in ["multi_concept", "single_concept", "only_question"]
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        statics_path = os.path.join(preprocessed_dir, f"statics_preprocessed_{data_type}.json")
        write_json(statics, statics_path)

    def save_data_statics_raw(self, statics, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        statics_path = os.path.join(preprocessed_dir, f"statics_raw.json")
        write_json(statics, statics_path)

    def save_data_id_map(self, all_id_maps, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        for data_type, id_maps in all_id_maps.items():
            for id_map_name, id_map in id_maps.items():
                id_map_path = os.path.join(preprocessed_dir, f"{id_map_name}_{data_type}.csv")
                id_map.to_csv(id_map_path, index=False)

    def get_q_table(self, dataset_name, data_type):
        assert data_type in ["multi_concept", "single_concept", "only_question"]
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        if data_type == "only_question":
            Q_table_path = os.path.join(preprocessed_dir, "Q_table_multi_concept.npy")
        else:
            Q_table_path = os.path.join(preprocessed_dir, f"Q_table_{data_type}.npy")
        try:
            Q_table = np.load(Q_table_path)
        except FileNotFoundError:
            Q_table = None
        return Q_table

    def get_data_statics_processed(self, dataset_name, data_type):
        assert data_type in ["multi_concept", "single_concept", "only_question"]
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        statics_path = os.path.join(preprocessed_dir, f"statics_preprocessed_{data_type}.json")
        statics = load_json(statics_path)
        return statics

    def get_preprocessed_path(self, dataset_name, data_type):
        assert data_type in ["multi_concept", "single_concept", "only_question"]
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        return os.path.join(preprocessed_dir, f"data_{data_type}.txt")

    def get_concept_id2name(self, dataset_name):
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        concept_id2name_path = os.path.join(preprocessed_dir, "concept_id2name_map.csv")
        return load_csv(concept_id2name_path)

    def get_concept_id_map(self, dataset_name, data_type):
        assert data_type in ["multi_concept", "single_concept"]
        preprocessed_dir = self.get_preprocessed_dir(dataset_name)
        concept_id_map_path = os.path.join(preprocessed_dir, f"concept_id_map_{data_type}.csv")
        return load_csv(concept_id_map_path)

    # ==================================================================================================================

    def get_models_dir(self):
        return os.path.join(self.root_dir, FileManager.models_dir_in_lab)

    # ==================================================================================================================
    def add_new_setting(self, setting_name, setting_info):
        setting_dir = os.path.join(self.root_dir, FileManager.setting_dir_in_lab, setting_name)
        if os.path.exists(setting_dir) and os.path.isdir(setting_dir):
            return
        setting_path = os.path.join(setting_dir, FileManager.file_settings_name)
        os.mkdir(setting_dir)
        with open(setting_path, "w") as f:
            json.dump(setting_info, f, indent=2)

    def delete_old_setting(self, setting_old_name):
        setting_dir = os.path.join(self.root_dir, FileManager.setting_dir_in_lab, setting_old_name)
        assert os.path.exists(setting_dir) or os.path.isdir(setting_dir), f"{setting_old_name} dir does not exist"
        os.rmdir(setting_dir)

    def get_setting_dir(self, setting_name):
        result_dir = os.path.join(self.root_dir, FileManager.setting_dir_in_lab)
        setting_names = os.listdir(result_dir)
        assert setting_name in setting_names, f"{setting_name} dir does not exist"
        return os.path.join(result_dir, setting_name)

    def get_setting_file_path(self, setting_name):
        setting_dir = self.get_setting_dir(setting_name)
        return os.path.join(setting_dir, FileManager.file_settings_name)
    # ==================================================================================================================
