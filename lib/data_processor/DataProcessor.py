import os
import re
import time
import pandas as pd

from copy import deepcopy


from . import CONSTANT
from . import load_raw
from . import util
from . import preprocess_raw
from ..util import parse


class DataProcessor:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

        # 数据集的一些统计信息
        self.statics_raw = None
        self.statics_preprocessed = {}

        # 处理后的数据
        self.data_raw = None
        self.data_preprocessed = {}
        self.Q_table = {}

        self.question_id_map = None
        self.concept_id_map = None
        self.school_info = None
        self.school_id_map = None
        self.country_info = None
        self.country_id_map = None

        # 统一格式的数据
        self.data_uniformed = {}

        # 一些数据的信息，如学校、城市
        self.other_info = {}

    def process_data(self):
        datasets_treatable = CONSTANT.datasets_treatable()
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        assert dataset_name in datasets_treatable, f"DataProcessor can't handle {dataset_name}"

        data_path = self.params["preprocess_config"]["data_path"]
        assert os.path.exists(data_path), f"raw data ({data_path}) not exist"

        if dataset_name in ["assist2009", "assist2009-new"]:
            self.process_assist2009()
        elif dataset_name == "assist2012":
            self.process_assist2012()
        else:
            raise NotImplementedError()

        self.uniform_data()
        return self.data_uniformed

    def process_assist2009(self):
        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # if dataset_name == "assist2015":
        #     df["question_id"] = df["question_id"].map(int)
        #
        # if dataset_name in ["assist2009", "assist2009-new", "assist2012", "assist2017"]:
        #     df["question_id"] = df["question_id"].map(int)
        #     df["concept_id"] = df["concept_id"].map(int)
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        df["question_id"] = df["question_id"].map(int)
        df["concept_id"] = df["concept_id"].map(int)
        result_preprocessed = preprocess_raw.preprocess_assist(dataset_name, df)

        self.data_preprocessed["multi_concept"] = result_preprocessed["multi_concept"]["data_processed"]
        self.data_preprocessed["single_concept"] = result_preprocessed["single_concept"]["data_processed"]
        self.Q_table["multi_concept"] = result_preprocessed["multi_concept"]["Q_table"]
        self.Q_table["single_concept"] = result_preprocessed["single_concept"]["Q_table"]
        self.statics_preprocessed["multi_concept"] = (
            self.get_basic_info(result_preprocessed["multi_concept"]["data_processed"]))
        self.statics_preprocessed["multi_concept"]["num_max_concept"] = (
            int(max(result_preprocessed["multi_concept"]["Q_table"].sum(axis=1))))
        self.statics_preprocessed["single_concept"] = (
            self.get_basic_info(result_preprocessed["single_concept"]["data_processed"]))

    def process_assist2012(self):
        def time_str2timestamp(time_str):
            if len(time_str) != 19:
                time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        df["correct"] = df["correct"].astype('int8')
        df["use_time"] = df["use_time"].map(lambda t: min(max(1, int(t) // 1000), 60 * 60))
        df["timestamp"] = df["timestamp"].map(time_str2timestamp)
        df["question_id"] = df["question_id"].map(int)
        df["concept_id"] = df["concept_id"].map(int)
        result_preprocessed = preprocess_raw.preprocess_assist(dataset_name, df)

        self.data_preprocessed["single_concept"] = result_preprocessed["single_concept"]["data_processed"]
        self.Q_table["single_concept"] = result_preprocessed["single_concept"]["Q_table"]
        self.statics_preprocessed["single_concept"] = (
            self.get_basic_info(result_preprocessed["single_concept"]["data_processed"]))

    def process_assist2015(self):
        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        df["question_id"] = df["question_id"].map(int)
        # result_preprocessed = preprocess_raw.preprocess_assist(dataset_name, df)

    def process_assist2017(self):
        pass

    def uniform_data(self):
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        if dataset_name in ["assist2009", "assist2009-new"]:
            self.uniform_assist2009()
        if dataset_name in ["assist2012", "assist2017"]:
            self.uniform_assist2012()
        pass

    def uniform_assist2009(self):
        df = deepcopy(self.data_preprocessed["multi_concept"])
        # school_id按照学生数量重映射
        df["school_id"] = df["school_id"].fillna(-1)
        df["school_id"] = df["school_id"].map(int)
        school_id_map, school_info = preprocess_raw.map_user_info(df, "school_id")

        info_name_table = {
            "question_seq": "question_id",
            "concept_seq": "concept_id",
            "correct_seq": "correct"
        }

        # only_question
        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"order_id"})
        dataset_seq_keys = CONSTANT.datasets_seq_keys()["assist2009"]
        # 去除多知识点习题的冗余
        df = df[~df.duplicated(subset=["user_id", "order_id", "question_id"])]
        # 多知识点数据集先生成single（只有习题id，无知识点id），再通过single和Q table扩展为multi
        dataset_info_names = list(set(dataset_seq_keys) - {"concept_seq"})
        seqs = []
        for user_id in pd.unique(df["user_id"]):
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["order_id"])
            object_data = {info_name: [] for info_name in dataset_info_names}
            for k in id_keys:
                object_data[k] = user_data.iloc[0][k]
            for i, (_, row_data) in enumerate(user_data.iterrows()):
                for info_name in dataset_info_names:
                    object_data[info_name].append(row_data[info_name_table[info_name]])
            object_data["seq_len"] = len(object_data["correct_seq"])
            seqs.append(object_data)
        self.data_uniformed["only_question"] = list(filter(lambda item: 2 <= item["seq_len"], seqs))

        # multi_concept
        seqs = self.single2multi(seqs, self.Q_table["multi_concept"])
        self.data_uniformed["multi_concept"] = list(filter(lambda item: 2 <= item["seq_len"], seqs))

        # single_concept
        df = self.data_preprocessed["single_concept"]
        seqs = []
        for user_id in pd.unique(df["user_id"]):
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["order_id"])
            object_data = {info_name: [] for info_name in dataset_seq_keys}
            for k in id_keys:
                object_data[k] = user_data.iloc[0][k]
            for i, (_, row_data) in enumerate(user_data.iterrows()):
                for info_name in dataset_seq_keys:
                    object_data[info_name].append(row_data[info_name_table[info_name]])
            object_data["seq_len"] = len(object_data["correct_seq"])
            seqs.append(object_data)
        self.data_uniformed["single_concept"] = list(filter(lambda item: 2 <= item["seq_len"], seqs))

    def uniform_assist2012(self):
        df = deepcopy(self.data_preprocessed["single_concept"])
        # school_id按照学生数量重映射
        df["school_id"] = df["school_id"].fillna(-1)
        df["school_id"] = df["school_id"].map(int)
        school_id_map, school_info = preprocess_raw.map_user_info(df, "school_id")

        info_name_table = {
            "question_seq": "question_id",
            "concept_seq": "concept_id",
            "correct_seq": "correct",
            "time_seq": "timestamp",
            "use_time_seq": "use_time"
        }

        # single_concept
        id_keys = list(set(df.columns) - set(info_name_table.values()))
        dataset_seq_keys = CONSTANT.datasets_seq_keys()["assist2012"]
        seqs = []
        for user_id in pd.unique(df["user_id"]):
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["timestamp"])
            object_data = {info_name: [] for info_name in dataset_seq_keys}
            for k in id_keys:
                object_data[k] = user_data.iloc[0][k]
            for i, (_, row_data) in enumerate(user_data.iterrows()):
                for info_name in dataset_seq_keys:
                    object_data[info_name].append(row_data[info_name_table[info_name]])
            object_data["seq_len"] = len(object_data["correct_seq"])
            seqs.append(object_data)
        self.data_uniformed["single_concept"] = list(filter(lambda item: 2 <= item["seq_len"], seqs))

    def uniform_assist2015(self):
        pass

    @staticmethod
    def get_basic_info(df):
        useful_cols = {"question_id", "concept_id", "concept_name", "school_id", "teacher_id", "age"}
        useful_cols = useful_cols.intersection(set(df.columns))
        result = {
            "num_interaction": len(df),
            "num_user": len(pd.unique(df["user_id"]))
        }
        for col in useful_cols:
            if col == "question_id":
                result["num_question"] = util.get_info_function(df, "question_id")
            elif col == "concept_id":
                result["num_concept"] = util.get_info_function(df, "concept_id")
            elif col == "concept_name":
                result["num_concept_name"] = util.get_info_function(df, "concept_name")
            elif col == "school_id":
                result["num_school"] = util.get_info_function(df, "school_id")
            elif col == "teacher_id":
                result["num_teacher"] = util.get_info_function(df, "teacher_id")
            elif col == "age":
                result["num_age"] = util.get_info_function(df, "age")
        return result

    @staticmethod
    def single2multi(seqs, Q_table):
        id_keys, seq_keys = parse.get_keys_from_uniform(seqs)
        seq_keys = list(set(seq_keys) - {"question_seq"})
        all_keys = id_keys + seq_keys

        seqs_new = []
        for item_data in seqs:
            item_data_new = {k: ([] if (k in seq_keys) else item_data[k]) for k in all_keys}
            item_data_new["question_seq"] = []
            item_data_new["concept_seq"] = []
            question_seq = item_data["question_seq"]
            # 这样第一个数据就是习题id
            seq_all = zip(question_seq, *(item_data[info_name] for info_name in seq_keys))
            for ele_all in seq_all:
                q_id = ele_all[0]
                c_ids = parse.get_concept_from_question(q_id, Q_table)
                len_c_ids = len(c_ids)
                item_data_new["question_seq"] += [q_id] + [-1] * (len_c_ids - 1)
                item_data_new["concept_seq"] += c_ids
                for i, info_name in enumerate(seq_keys):
                    item_data_new[info_name] += [ele_all[i + 1]] * len_c_ids
            item_data_new["seq_len"] = len(item_data_new["correct_seq"])
            seqs_new.append(item_data_new)
        return seqs_new
