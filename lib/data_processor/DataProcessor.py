import os
import re
import time
import datetime
import pandas as pd
import numpy as np

from copy import deepcopy


from . import CONSTANT
from . import load_raw
from . import util
from . import preprocess_raw
from ..util import parse as parse_util
from ..util import data as data_util


class DataProcessor:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

        self.data_raw = None
        self.statics_raw = None
        # {} 表示数据有三种形式multi_concept, single_concept, only_question
        self.data_preprocessed = {}
        self.statics_preprocessed = {}
        self.Q_table = {}
        self.question_id_map = {}
        self.concept_id_map = {}

        self.school_info = None
        self.school_id_map = None
        self.country_info = None
        self.country_id_map = None

        # 统一格式的数据
        self.data_uniformed = {}

        # 一些数据的信息，如学校、城市
        self.other_info = {}

    def process_data(self):
        datasets_treatable = self.objects["file_manager"].builtin_datasets
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        assert dataset_name in datasets_treatable, f"DataProcessor can't handle {dataset_name}"

        data_path = self.params["preprocess_config"]["data_path"]
        assert os.path.exists(data_path), f"raw data ({data_path}) not exist"

        if dataset_name == "assist2009":
            self.load_process_assist2009()
        elif dataset_name == "assist2009-full":
            self.load_process_assist2009_full()
        elif dataset_name == "assist2012":
            self.load_process_assist2012()
        elif dataset_name == "assist2015":
            self.load_process_assist2015()
        elif dataset_name == "assist2017":
            self.load_process_assist2017()
        elif dataset_name == "edi2020-task1":
            self.load_process_edi2020_task1()
        elif dataset_name == "edi2020-task34":
            self.load_process_edi2020_task34()
        elif dataset_name == "ednet-kt1":
            self.load_process_ednet_kt1()
        elif dataset_name == "xes3g5m":
            self.load_process_uniform_xes3g5m()
        elif dataset_name in ["algebra2005", "algebra2006", "algebra2008", "bridge2algebra2006", "bridge2algebra2008"]:
            self.load_process_kdd_cup2010()
        elif dataset_name in ["SLP-bio", "SLP-chi", "SLP-eng", "SLP-geo", "SLP-his", "SLP-mat", "SLP-phy"]:
            self.load_process_SLP()
        elif dataset_name == "statics2011":
            self.load_process_statics2011()
        else:
            raise NotImplementedError()

        self.uniform_data()
        return self.data_uniformed

    def load_process_assist2009(self):
        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = "assist2009"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # 有知识点名称的interaction数量为325637，总数量为401756
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        df["question_id"] = df["question_id"].map(int)
        df["concept_id"] = df["concept_id"].map(int)

        # 获取concept name和concept 原始id的对应并保存
        concept_names = list(pd.unique(df.dropna(subset=["concept_name"])["concept_name"]))
        concept_id2name = {}
        for c_name in concept_names:
            concept_data = df[df["concept_name"] == c_name]
            c_id = int(concept_data["concept_id"].iloc[0])
            concept_id2name[c_id] = c_name.strip()
        concept_id2name_map = pd.DataFrame({
            "concept_id": concept_id2name.keys(),
            "concept_name": concept_id2name.values()
        })
        preprocessed_dir = self.objects["file_manager"].get_preprocessed_dir(dataset_name)
        concept_id2name_map_path = os.path.join(preprocessed_dir, f"concept_id2name_map.csv")
        concept_id2name_map.to_csv(concept_id2name_map_path, index=False)

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
        self.question_id_map["multi_concept"] = result_preprocessed["multi_concept"]["question_id_map"]
        self.concept_id_map["multi_concept"] = result_preprocessed["multi_concept"]["concept_id_map"]
        self.question_id_map["single_concept"] = result_preprocessed["single_concept"]["question_id_map"]
        self.concept_id_map["single_concept"] = result_preprocessed["single_concept"]["concept_id_map"]

    def load_process_assist2009_full(self):
        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = "assist2009-full"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        # single_concept
        df = deepcopy(self.data_raw)
        df.dropna(subset=["user_id", "question_id", "concept_id"], inplace=True)
        df_multi_concept = deepcopy(self.data_raw)
        df_multi_concept.dropna(subset=["user_id", "question_id", "concept_id"], inplace=True)
        self.process_raw_is_single_concept(df, df_multi_concept, c_id_is_num=True)

    def load_process_assist2012(self):
        def time_str2timestamp(time_str):
            if len(time_str) != 19:
                time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = "assist2012"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # 有知识点名称的interaction数量为2630080，总数量为2711813
        df.dropna(subset=["question_id", "concept_id"], inplace=True)

        # 获取concept name和concept 原始id的对应并保存
        concept_names = list(pd.unique(df.dropna(subset=["concept_name"])["concept_name"]))
        concept_id2name = {}
        for c_name in concept_names:
            concept_data = df[df["concept_name"] == c_name]
            c_id = int(concept_data["concept_id"].iloc[0])
            concept_id2name[c_id] = c_name.strip()
        concept_id2name_map = pd.DataFrame({
            "concept_id": concept_id2name.keys(),
            "concept_name": concept_id2name.values()
        })
        preprocessed_dir = self.objects["file_manager"].get_preprocessed_dir(dataset_name)
        concept_id2name_map_path = os.path.join(preprocessed_dir, f"concept_id2name_map.csv")
        concept_id2name_map.to_csv(concept_id2name_map_path, index=False)

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
        self.question_id_map["single_concept"] = result_preprocessed["single_concept"]["question_id_map"]
        self.concept_id_map["single_concept"] = result_preprocessed["single_concept"]["concept_id_map"]

    def load_process_assist2015(self):
        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = "assist2015"
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, rename_dict=rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # 常规处理，先丢弃correct不为0或1的数据（可能为小数）
        df = df[(df["correct"] == 0) | (df["correct"] == 1)]
        df["correct"] = df["correct"].map(int)
        df["question_id"] = df["question_id"].map(int)
        question_ids = pd.unique(df["question_id"])
        df["question_id"] = df["question_id"].map({c_id: i for i, c_id in enumerate(question_ids)})
        self.data_preprocessed["only_question"] = df
        self.statics_preprocessed["only_question"] = DataProcessor.get_basic_info(df)

        question_info = pd.DataFrame({
            "question_id": question_ids,
            "question_mapped_id": range(len(question_ids))
        })
        self.question_id_map["only_question"] = question_info

    def load_process_assist2017(self):
        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = "assist2017"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        # skill字段有noskill值，过滤掉
        df = deepcopy(self.data_raw[self.data_raw["concept_id"] != "noskill"])
        df["use_time"] = df["use_time"].map(lambda t: min(max(1, int(t)), 60 * 60))
        skill_id_map = {}
        skill_names = pd.unique(df["concept_id"])
        for i, skill in enumerate(skill_names):
            skill_id_map[skill] = i
        skill_id_column = df.apply(lambda l: skill_id_map[l["concept_id"]], axis=1)
        df["concept_id"] = skill_id_column

        # 里面有些习题id对应多个知识点id（但实际不是同一个习题），对这些习题id进行重映射，使之成为单知识点数据集
        question_concept_pairs = {}
        for i in df.index:
            question_concept_pair = str(int(df["question_id"][i])) + "," + str(int(df["concept_id"][i]))
            question_concept_pairs.setdefault(question_concept_pair, len(question_concept_pairs))
            df["question_id"][i] = question_concept_pairs[question_concept_pair]

        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        df["question_id"] = df["question_id"].map(int)
        df["concept_id"] = df["concept_id"].map(int)
        result_preprocessed = preprocess_raw.preprocess_assist(dataset_name, df)

        self.data_preprocessed["single_concept"] = result_preprocessed["single_concept"]["data_processed"]
        self.Q_table["single_concept"] = result_preprocessed["single_concept"]["Q_table"]
        self.statics_preprocessed["single_concept"] = (
            self.get_basic_info(result_preprocessed["single_concept"]["data_processed"]))
        # assist2017的习题id经过了两次映射
        qc_pairs_reverse = {v: int(k.split(",")[0]) for k, v in question_concept_pairs.items()}
        self.question_id_map["single_concept"] = result_preprocessed["single_concept"]["question_id_map"]
        self.question_id_map["single_concept"]["question_id"] = self.question_id_map["single_concept"]["question_id"].map(qc_pairs_reverse)
        self.concept_id_map["single_concept"] = result_preprocessed["single_concept"]["concept_id_map"]

    def load_process_edi2020_task1(self):
        data_dir = self.params["preprocess_config"]["data_path"]
        data_train_path = os.path.join(data_dir, "train_data", "train_task_1_2.csv")
        task1_test_public_path = os.path.join(data_dir, "test_data", "test_public_answers_task_1.csv")
        task1_test_private_path = os.path.join(data_dir, "test_data", "test_private_answers_task_1.csv")
        metadata_answer_path = os.path.join(data_dir, "metadata", "answer_metadata_task_1_2.csv")
        metadata_question_path = os.path.join(data_dir, "metadata", "question_metadata_task_1_2.csv")
        metadata_student_path = os.path.join(data_dir, "metadata", "student_metadata_task_1_2.csv")

        metadata_answer_cols = ["AnswerId", "DateAnswered"]
        df_cols = ["QuestionId", "AnswerId", "UserId", "IsCorrect"]
        df_rename_map = {
            "QuestionId": "question_id",
            "AnswerId": "answer_id",
            "UserId": "user_id",
            "IsCorrect": "correct"
        }
        meta_question_rename_map = {
            "QuestionId": "question_id",
            "SubjectId": "concept_ids"
        }
        meta_student_raname_map = {
            "UserId": "user_id",
            "Gender": "gender",
            "DateOfBirth": "birth",
            "PremiumPupil": "premium_pupil"
        }
        meta_answer_rename_map = {
            "AnswerId": "answer_id",
            "DateAnswered": "timestamp"
        }

        # df_train、df_test_public、df_test_private均无NaN
        # metadata_student在DateOfBirth和PremiumPupil列有NaN
        # metadata_question无NaN
        # metadata_answer在AnswerId、Confidence和SchemeOfWorkId列有NaN
        df_train = load_raw.load_csv(data_train_path, df_cols, df_rename_map)
        df_task1_test_public = load_raw.load_csv(task1_test_public_path, df_cols, df_rename_map)
        df_task1_test_private = load_raw.load_csv(task1_test_private_path, df_cols, df_rename_map)
        metadata_answer = load_raw.load_csv(metadata_answer_path, metadata_answer_cols, meta_answer_rename_map)
        metadata_question = load_raw.load_csv(metadata_question_path, rename_dict=meta_question_rename_map)
        metadata_student = load_raw.load_csv(metadata_student_path, rename_dict=meta_student_raname_map)

        # 0,1,2分别表示train、test pub、test pri
        df_train.insert(loc=len(df_train.columns), column='dataset_type', value=0)
        df_task1_test_public.insert(loc=len(df_task1_test_public.columns), column='dataset_type', value=1)
        df_task1_test_private.insert(loc=len(df_task1_test_private.columns), column='dataset_type', value=2)
        self.data_raw = pd.concat((
            df_train,
            df_task1_test_public,
            df_task1_test_private
        ), axis=0)
        self.statics_raw = DataProcessor.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        metadata_answer.dropna(subset=["answer_id"], inplace=True)
        metadata_answer["answer_id"] = metadata_answer["answer_id"].map(int)
        metadata_student["premium_pupil"] = metadata_student["premium_pupil"].fillna(2)
        metadata_student["premium_pupil"] = metadata_student["premium_pupil"].map(int)

        def time_str2year(time_str):
            if str(time_str) == "nan":
                # 后面用time_year - birth，如果为0或者为负数，说明至少其中一项为NaN
                return 3000
            return int(time_str[:4])

        def time_str2timestamp(time_str):
            # if len(time_str) != 19:
            #     time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        def map_gender(item):
            if item not in [1, 2]:
                return 0
            return item

        time_year = metadata_answer["timestamp"].map(time_str2year)
        time_year.name = "time_year"
        metadata_answer["timestamp"] = metadata_answer["timestamp"].map(time_str2timestamp)
        metadata_answer = pd.concat((metadata_answer, time_year), axis=1)
        metadata_student["birth"] = metadata_student["birth"].map(time_str2year)
        df = df.merge(metadata_student, how="left")
        df["gender"] = df["gender"].map(map_gender)

        # 丢弃没有时间戳的数据
        metadata_answer = metadata_answer[~metadata_answer.duplicated(subset=["answer_id"])]
        df = df.merge(metadata_answer, how="left")
        df.dropna(subset=["timestamp"], inplace=True)

        # 连接metadata_question
        df = df.merge(metadata_question, how="left")
        df.dropna(subset=["concept_ids"], inplace=True)

        # 习题id重映射
        question_ids = list(pd.unique(df["question_id"]))
        question_ids.sort()
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        question_info = pd.DataFrame({
            "question_id": question_ids,
            "question_mapped_id": range(len(question_ids))
        })

        # 计算年龄：time_year - birth
        age = df["time_year"] - df["birth"]
        age.name = "age"
        age[(age <= 6) | (age > 40)] = 0
        age[(age > 20) & (age <= 40)] = 21
        df = pd.concat([df, age], axis=1)

        # 知识点id重映射，metadata_question中Subject_Id是层级知识点，取最后一个（最细粒度的）知识点作为习题知识点，所以是单知识点数据集
        concept_ids = set()
        question_concept_map = {}
        for i in range(len(metadata_question)):
            q_id = metadata_question.iloc[i]["question_id"]
            c_ids_str = metadata_question.iloc[i]["concept_ids"]
            c_ids = eval(c_ids_str)
            question_concept_map[question_id_map[q_id]] = [c_ids[-1]]
            concept_ids.add(c_ids[-1])
        concept_ids = list(concept_ids)
        concept_ids.sort()
        concept_info = pd.DataFrame({
            "concept_id": concept_ids,
            "concept_mapped_id": range(len(concept_ids))
        })
        self.question_id_map["single_concept"] = question_info
        self.concept_id_map["single_concept"] = concept_info

        # 习题到知识点的映射
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        for q_id in question_concept_map.keys():
            question_concept_map[q_id] = list(map(lambda c_id: concept_id_map[c_id], question_concept_map[q_id]))
        df_question_concept = pd.DataFrame({
            "question_id": question_concept_map.keys(),
            "concept_id": map(lambda c_ids_: c_ids_[0], question_concept_map.values())
        })
        df = df.merge(df_question_concept, how="left", on=["question_id"])
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        self.data_preprocessed["single_concept"] = df[
            ["user_id", "question_id", "concept_id", "correct", "age", "timestamp", "gender", "premium_pupil", "dataset_type"]
        ]
        self.statics_preprocessed["single_concept"] = (
            DataProcessor.get_basic_info(self.data_preprocessed["single_concept"])
        )

        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for q_id in question_concept_map.keys():
            correspond_c = question_concept_map[q_id]
            Q_table[[q_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table["single_concept"] = Q_table

    def load_process_edi2020_task34(self):
        data_dir = self.params["preprocess_config"]["data_path"]
        data_train_path = os.path.join(data_dir, "train_data", "train_task_3_4.csv")
        metadata_answer_path = os.path.join(data_dir, "metadata", "answer_metadata_task_3_4.csv")
        metadata_question_path = os.path.join(data_dir, "metadata", "question_metadata_task_3_4.csv")
        metadata_student_path = os.path.join(data_dir, "metadata", "student_metadata_task_3_4.csv")

        # df_train、df_test_public、df_test_private均无NaN
        # metadata_student在DateOfBirth和PremiumPupil列有NaN
        # metadata_question无NaN
        # metadata_answer在AnswerId、Confidence和SchemeOfWorkId列有NaN
        df_rename_map = {
            "QuestionId": "question_id",
            "AnswerId": "answer_id",
            "UserId": "user_id",
            "IsCorrect": "correct"
        }
        meta_question_rename_map = {
            "QuestionId": "question_id",
            "SubjectId": "concept_ids"
        }
        meta_student_raname_map = {
            "UserId": "user_id",
            "Gender": "gender",
            "DateOfBirth": "birth",
            "PremiumPupil": "premium_pupil"
        }
        meta_answer_rename_map = {
            "AnswerId": "answer_id",
            "DateAnswered": "timestamp"
        }

        df = load_raw.load_csv(data_train_path, useful_cols=["QuestionId", "AnswerId", "UserId", "IsCorrect"],
                               rename_dict=df_rename_map)
        metadata_answer = load_raw.load_csv(metadata_answer_path, useful_cols=["AnswerId", "DateAnswered"],
                                            rename_dict=meta_answer_rename_map)
        metadata_question = load_raw.load_csv(metadata_question_path, rename_dict=meta_question_rename_map)
        metadata_student = load_raw.load_csv(metadata_student_path, rename_dict=meta_student_raname_map)

        # 对这个数据集，只使用比赛数据的训练集，在pykt-toolkit框架中就是这么处理的
        metadata_answer.dropna(subset=["answer_id"], inplace=True)
        metadata_answer["answer_id"] = metadata_answer["answer_id"].map(int)
        metadata_student["premium_pupil"] = metadata_student["premium_pupil"].fillna(-1)
        metadata_student["premium_pupil"] = metadata_student["premium_pupil"].map(int)

        def time_str2year(time_str):
            if str(time_str) == "nan":
                # 后面用time_year - birth，如果为0或者为负数，说明至少其中一项为NaN
                return 3000
            return int(time_str[:4])

        def time_str2timestamp(time_str):
            # if len(time_str) != 19:
            #     time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        def map_gender(item):
            # if item not in [1, 2]:
            #     return 0
            if item == 1:
                return 0
            elif item == 2:
                return 1
            else:
                return -1

        time_year = metadata_answer["timestamp"].map(time_str2year)
        time_year.name = "time_year"
        metadata_answer["timestamp"] = metadata_answer["timestamp"].map(time_str2timestamp)
        metadata_answer = pd.concat((metadata_answer, time_year), axis=1)
        metadata_student["birth"] = metadata_student["birth"].map(time_str2year)
        df = df.merge(metadata_student, how="left")
        df["gender"] = df["gender"].map(map_gender)

        # 丢弃没有时间戳的数据
        metadata_answer = metadata_answer[~metadata_answer.duplicated(subset=["answer_id"])]
        df = df.merge(metadata_answer, how="left")
        df.dropna(subset=["timestamp"], inplace=True)

        # 连接metadata_question
        df = df.merge(metadata_question, how="left")
        df.dropna(subset=["concept_ids"], inplace=True)

        # 习题id重映射
        question_ids = list(pd.unique(df["question_id"]))
        question_ids.sort()
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        question_info = pd.DataFrame({
            "question_id": question_ids,
            "question_mapped_id": range(len(question_ids))
        })

        # 计算年龄：time_year - birth
        age = df["time_year"] - df["birth"]
        age.name = "age"
        age[(age <= 6) | (age > 40)] = -1
        age[(age > 20) & (age <= 40)] = 21
        df = pd.concat([df, age], axis=1)

        # 知识点id重映射，metadata_question中Subject_Id是层级知识点，取最后一个（最细粒度的）知识点作为习题知识点，所以是单知识点数据集
        concept_ids = set()
        question_concept_map = {}
        for i in range(len(metadata_question)):
            q_id = metadata_question.iloc[i]["question_id"]
            c_ids_str = metadata_question.iloc[i]["concept_ids"]
            c_ids = eval(c_ids_str)
            question_concept_map[question_id_map[q_id]] = [c_ids[-1]]
            concept_ids.add(c_ids[-1])
        concept_ids = list(concept_ids)
        concept_ids.sort()
        concept_info = pd.DataFrame({
            "concept_id": concept_ids,
            "concept_mapped_id": range(len(concept_ids))
        })
        self.question_id_map["single_concept"] = question_info
        self.concept_id_map["single_concept"] = concept_info

        # 习题到知识点的映射
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        for q_id in question_concept_map.keys():
            question_concept_map[q_id] = list(map(lambda c_id: concept_id_map[c_id], question_concept_map[q_id]))
        df_question_concept = pd.DataFrame({
            "question_id": question_concept_map.keys(),
            "concept_id": map(lambda c_ids_: c_ids_[0], question_concept_map.values())
        })
        df = df.merge(df_question_concept, how="left", on=["question_id"])
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        self.data_preprocessed["single_concept"] = df[["user_id", "question_id", "concept_id", "correct", "age",
                                                       "timestamp", "gender", "premium_pupil"]]
        self.statics_preprocessed["single_concept"] = (
            DataProcessor.get_basic_info(self.data_preprocessed["single_concept"]))

        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for q_id in question_concept_map.keys():
            correspond_c = question_concept_map[q_id]
            Q_table[[q_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table["single_concept"] = Q_table

    def load_process_ednet_kt1(self):
        data_dir = self.params["preprocess_config"]["data_path"]
        # ednet-kt1共有72000多名学生的记录，通常的处理是随机取5000名学生记录
        # data_dir下每个文件存放了5000名学生（随机但每个文件下学生不重复）的记录，num_file指定要读取几个文件，按照常规处理就取1
        self.data_raw = load_raw.load_ednet_kt1(data_dir, num_file=1)
        dataset_name = "ednet-kt1"
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw.rename(columns=rename_cols, inplace=True)

        # single_concept
        df = deepcopy(self.data_raw)
        df["use_time"] = df["use_time"].map(lambda t: min(max(1, int(t) // 1000), 60 * 60))
        df_multi_concept = deepcopy(self.data_raw)
        self.process_raw_is_single_concept(df, df_multi_concept, c_id_is_num=True)

    def load_process_uniform_xes3g5m(self):
        # 习题类型：单选和填空
        data_dir = self.params["preprocess_config"]["data_path"]
        # kc_level和question_level的数据是一样的，前者是multi_concept，后者是only_question（对于多知识点习题用 _ 拼接知识点）
        train_valid_path = os.path.join(data_dir, "question_level", "train_valid_sequences_quelevel.csv")
        test_path = os.path.join(data_dir, "question_level", "test_quelevel.csv")
        df_train_valid = load_raw.load_csv(train_valid_path)[
            ["uid", "questions", "concepts", "responses", "timestamps", "selectmasks"]
        ]
        df_test = load_raw.load_csv(test_path)[["uid", "questions", "concepts", "responses", "timestamps"]]

        # metadata
        question_meta_path = os.path.join(data_dir, "metadata", "questions.json")
        concept_meta_path = os.path.join(data_dir, "metadata", "kc_routes_map.json")
        question_meta = data_util.load_json(question_meta_path)
        concept_meta = data_util.load_json(concept_meta_path)

        concept_meta = {int(c_id): c_name.strip() for c_id, c_name in concept_meta.items()}
        question_meta = {int(q_id): q_meta for q_id, q_meta in question_meta.items()}
        data_processed_dir = self.objects["file_manager"].get_preprocessed_dir("xes3g5m")
        question_meta_path = os.path.join(data_processed_dir, "question_meta.json")
        concept_meta_path = os.path.join(data_processed_dir, "concept_meta.json")
        data_util.write_json(concept_meta, concept_meta_path)
        data_util.write_json(question_meta, question_meta_path)

        data_all = {}
        for i in df_train_valid.index:
            user_id = int(df_train_valid["uid"][i])
            data_all.setdefault(user_id, {
                "question_seq": [],
                "concept_seq": [],
                "correct_seq": [],
                "time_seq": []
            })
            # df_train_valid提供的数据是切割好的（将长序列切成固定长度为200的序列），不足200的用-1补齐
            mask_seq = list(map(int, df_train_valid["selectmasks"][i].split(",")))
            if -1 in mask_seq:
                end_pos = mask_seq.index(-1)
            else:
                end_pos = 200

            question_seq = list(map(int, df_train_valid["questions"][i].split(",")))[:end_pos]
            concept_seq = list(map(lambda cs_str: list(map(int, cs_str.split("_"))),
                                   df_train_valid["concepts"][i].split(",")))[:end_pos]
            correct_seq = list(map(int, df_train_valid["responses"][i].split(",")))[:end_pos]
            time_seq = list(map(int, df_train_valid["timestamps"][i].split(",")))[:end_pos]
            data_all[user_id]["question_seq"] += question_seq
            data_all[user_id]["concept_seq"] += concept_seq
            data_all[user_id]["correct_seq"] += correct_seq
            data_all[user_id]["time_seq"] += time_seq

        for i in df_test.index:
            # df_test提供的数据是未切割的
            user_id = int(df_test["uid"][i])
            data_all.setdefault(user_id, {
                "question_seq": [],
                "concept_seq": [],
                "correct_seq": [],
                "time_seq": []
            })
            question_seq = list(map(int, df_test["questions"][i].split(",")))
            concept_seq = list(map(lambda cs_str: list(map(int, cs_str.split("_"))),
                                   df_test["concepts"][i].split(",")))
            correct_seq = list(map(int, df_test["responses"][i].split(",")))
            time_seq = list(map(int, df_test["timestamps"][i].split(",")))
            data_all[user_id]["question_seq"] += question_seq
            data_all[user_id]["concept_seq"] += concept_seq
            data_all[user_id]["correct_seq"] += correct_seq
            data_all[user_id]["time_seq"] += time_seq

        # 处理成统一格式，即[{user_id(int), question_seq(list), concept_seq(list), correct_seq(list), time_seq(list)}, ...]
        data_uniformed = [{
            "user_id": user_id,
            "question_seq": seqs["question_seq"],
            "concept_seq": seqs["concept_seq"],
            "correct_seq": seqs["correct_seq"],
            "time_seq": seqs["time_seq"],
            "seq_len": len(seqs["correct_seq"])
        } for user_id, seqs in data_all.items()]

        # 提取每道习题对应的知识点：提供的数据（train_valid_sequences_quelevel.csv和test_quelevel.csv）中习题对应的知识点是最细粒度的，类似edi2020数据集中层级知识点里最细粒度的知识点
        # 而question metadata里每道题的kc routes是完整的知识点（层级）
        # 并且提供的数据中习题对应知识点和question metadata中习题对应的知识点不是完全一一对应的，例如习题1035
        # 在question metadata中对应的知识点为
        # ['拓展思维----应用题模块----年龄问题----年龄问题基本关系----年龄差', '能力----运算求解',
        #  '课内题型----综合与实践----应用题----倍数问题----已知两量之间倍数关系和两量之差，求两个量',
        #  '学习能力----七大能力----运算求解',
        #  '拓展思维----应用题模块----年龄问题----年龄问题基本关系----年龄问题基本关系和差问题',
        #  '课内知识点----数与运算----数的运算的实际应用（应用题）----整数的简单实际问题----除法的实际应用',
        #  '知识点----应用题----和差倍应用题----已知两量之间倍数关系和两量之差，求两个量',
        #  '知识点----数的运算----估算与简单应用----整数的简单实际问题----除法的实际应用']
        # 在数据中对应的知识点为[169, 177, 239, 200, 73]，其对应的知识点名称为['除法的实际应用', '已知两量之间倍数关系和两量之差，求两个量', '年龄差', '年龄问题基本关系和差问题', '运算求解']
        question_ids = []
        concept_ids = []
        question_concept_map = {}
        for item_data in data_uniformed:
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                c_ids = item_data["concept_seq"][i]
                question_concept_map.setdefault(q_id, c_ids)
                question_ids.append(q_id)
                concept_ids.extend(c_ids)

        # 习题和知识点id都是映射过的，但是习题共有7651个，其id却是从0开始，7651结束（有一个空缺，但是不会影响后续模型训练，所以就不处理了）
        question_ids = sorted(list(set(question_ids)))
        concept_ids = sorted(list(set(concept_ids)))
        Q_table_multi_concept = np.zeros((len(question_ids) + 1, len(concept_ids)), dtype=int)
        for q_id in question_concept_map.keys():
            correspond_c = question_concept_map[q_id]
            Q_table_multi_concept[[q_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table["multi_concept"] = Q_table_multi_concept

        # 处理为multi_concept和only_question
        data_only_question = []
        for item_data in data_uniformed:
            item_data_only_question = {}
            for k in item_data:
                if k != "concept_seq":
                    item_data_only_question[k] = deepcopy(item_data[k])
            data_only_question.append(item_data_only_question)
        self.data_uniformed["only_question"] = data_only_question
        self.data_uniformed["multi_concept"] = DataProcessor.single2multi(data_only_question, Q_table_multi_concept)

        self.statics_preprocessed["multi_concept"] = {}
        self.statics_preprocessed["multi_concept"]["num_user"] = len(data_only_question)
        self.statics_preprocessed["multi_concept"]["num_interaction"] = (
            sum(list(map(lambda x: x["seq_len"], self.data_uniformed["multi_concept"]))))
        self.statics_preprocessed["multi_concept"]["num_concept"] = len(concept_ids)
        self.statics_preprocessed["multi_concept"]["num_question"] = len(question_ids)
        self.statics_preprocessed["multi_concept"]["num_max_concept"] = (int(max(Q_table_multi_concept.sum(axis=1))))

        # 处理为single_concept，即多知识点看成新的单知识点
        def process_c_ids(_c_ids):
            # 统一表示成id小的在前面，如多知识点组合[2,1,3]表示为"1_2_3"
            _c_ids_str = list(map(str, sorted(list(map(int, _c_ids)))))
            return "_".join(_c_ids_str)

        for item_data in data_uniformed:
            for i in range(item_data["seq_len"]):
                item_data["concept_seq"][i] = process_c_ids(item_data["concept_seq"][i])

        result_remap = util.id_remap4data_uniformed(data_uniformed, ["concept_seq"])
        self.data_uniformed["single_concept"] = result_remap[0]
        self.concept_id_map["single_concept"] = pd.DataFrame({
            "concept_id": result_remap[1]["concept_seq"].keys(),
            "concept_id_map": result_remap[1]["concept_seq"].values()
        })

        # 获取single concept的Q table
        question_ids = []
        concept_ids = []
        question_concept_map = {}
        for item_data in self.data_uniformed["single_concept"]:
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                c_id = item_data["concept_seq"][i]
                question_concept_map.setdefault(q_id, [c_id])
                question_ids.append(q_id)
                concept_ids.append(c_id)

        question_ids = sorted(list(set(question_ids)))
        concept_ids = sorted(list(set(concept_ids)))
        Q_table_single_concept = np.zeros((len(question_ids) + 1, len(concept_ids)), dtype=int)
        for q_id in question_concept_map.keys():
            correspond_c = question_concept_map[q_id]
            Q_table_single_concept[[q_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table["single_concept"] = Q_table_single_concept

        self.statics_preprocessed["single_concept"] = {}
        self.statics_preprocessed["single_concept"]["num_user"] = len(self.data_uniformed["single_concept"])
        self.statics_preprocessed["single_concept"]["num_interaction"] = (
            sum(list(map(lambda x: x["seq_len"], self.data_uniformed["single_concept"]))))
        self.statics_preprocessed["single_concept"]["num_concept"] = len(concept_ids)
        self.statics_preprocessed["single_concept"]["num_question"] = len(question_ids)

    def load_process_kdd_cup2010(self):
        def time_str2timestamp(time_str):
            if len(time_str) != 19:
                time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        dir_table = {
            "algebra2005": "algebra_2005_2006",
            "algebra2006": "algebra_2006_2007",
            "algebra2008": "algebra_2008_2009",
            "bridge2algebra2006": "bridge_to_algebra_2006_2007",
            "bridge2algebra2008": "bridge_to_algebra_2008_2009"
        }
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        data_path = self.params["preprocess_config"]["data_path"]
        train_path = os.path.join(data_path, f"{dir_table[dataset_name]}_train.txt")
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        data = load_raw.load_table(train_path, useful_cols, rename_cols)
        data["Problem Name"] = data["Problem Name"].apply(lambda x: x.strip().replace("_", "####").replace(",", "@@@@"))
        data["Step Name"] = data["Step Name"].apply(lambda x: x.strip().replace("_", "####").replace(",", "@@@@"))
        data["question_id"] = data.apply(lambda x: f"{x['Problem Name']}----{x['Step Name']}", axis=1)
        data = data[["user_id", "question_id", "concept_id", "correct", "timestamp"]]
        self.data_raw = data
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        df.dropna(subset=["user_id", "question_id", "concept_id", "timestamp", "correct"], inplace=True)
        df = df[df["correct"].isin([0, 1])]
        df["timestamp"] = df["timestamp"].map(time_str2timestamp)

        df_multi_concept = deepcopy(self.data_raw)
        df_multi_concept.dropna(subset=["user_id", "question_id", "concept_id", "timestamp", "correct"], inplace=True)
        df_multi_concept = df_multi_concept[df_multi_concept["correct"].isin([0, 1])]

        self.process_raw_is_single_concept(df, df_multi_concept, c_id_is_num=False)

    def load_process_SLP(self):
        data_dir = self.params["preprocess_config"]["data_path"]
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        self.data_raw = load_raw.load_SLP(data_dir, dataset_name)
        self.data_raw.rename(columns=CONSTANT.datasets_renamed()["SLP"], inplace=True)
        self.statics_raw = self.get_basic_info(self.data_raw)

        # 去除question_id和concept_id为nan以及"n.a."的数据
        df = deepcopy(self.data_raw)
        df.dropna(subset=["question_id", "concept_id", "score", "full_score", "timestamp"], inplace=True)
        df = df[(df["question_id"] != "n.a.") & (df["concept_id"] != "n.a.")]

        # 将user_id、question_id、concept_id、school_type、live_on_campus、timestamp映射为数字
        user_ids = list(pd.unique(df["user_id"]))
        df["user_id"] = df["user_id"].map({u_id: i for i, u_id in enumerate(user_ids)})

        question_ids = list(pd.unique(df["question_id"]))
        concept_ids = list(pd.unique(df["concept_id"]))
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        df["concept_id"] = df["concept_id"].map(concept_id_map)
        question_info = pd.DataFrame({
            "question_id": question_id_map.keys(),
            "question_mapped_id": question_id_map.values()
        })
        concept_info = pd.DataFrame({
            "concept_id": concept_id_map.keys(),
            "concept_mapped_id": concept_id_map.values()
        })
        self.question_id_map["single_concept"] = question_info
        self.concept_id_map["single_concept"] = concept_info

        def map_campus(item):
            item_str = str(item)
            if item_str == "Yes":
                return 1
            if item_str == "No":
                return 2
            return 0

        df["live_on_campus"] = df["live_on_campus"].map(map_campus)

        def map_gender(item):
            item_str = str(item)
            if item_str == "Male":
                return 2
            if item_str == "Female":
                return 1
            return 0

        df["gender"] = df["gender"].map(map_gender)

        def time_str2timestamp(time_str):
            if "-" in time_str:
                return int(time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S")))
            else:
                return int(time.mktime(time.strptime(time_str, "%Y/%m/%d %H:%M")))

        df["timestamp"] = df["timestamp"].map(time_str2timestamp)
        df["order"] = df["order"].map(int)

        # 将score和full_score转换为correct
        df["score"] = df["score"].map(float)

        def map_full_score(item):
            item_str = str(item)
            if item_str == "n.a.":
                return 1.0
            return float(item_str)

        df["full_score"] = df["full_score"].map(map_full_score)
        df["score_"] = df["score"] / df["full_score"]
        df["correct"] = df["score_"] > 0.5
        df["correct"] = df["correct"].map(int)
        df.rename(columns={"live_on_campus": "campus"}, inplace=True)
        self.data_preprocessed["single_concept"] = df[
            ["user_id", "question_id", "concept_id", "correct", "gender",
             "campus", "school_id", "timestamp", "order", "interaction_type"]
        ]
        self.statics_preprocessed["single_concept"] = (
            DataProcessor.get_basic_info(self.data_preprocessed["single_concept"])
        )

        # Q table
        df_new = pd.DataFrame({
            "question_id": map(int, df["question_id"].tolist()),
            "concept_id": map(int, df["concept_id"].tolist())
        })
        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
            correspond_c = pd.unique(group_info["concept_id"]).tolist()
            Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table["single_concept"] = Q_table

    def load_process_statics2011(self):
        data_path = self.params["preprocess_config"]["data_path"]
        dataset_name = "statics2011"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = load_raw.load_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)

        def replace_text(text):
            text = text.replace("_", "####").replace(",", "@@@@")
            return text

        def time_str2timestamp(time_str):
            datetime_obj = datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M")
            timestamp = int(datetime_obj.timestamp() / 60)
            return timestamp

        def process_concept_id(c_id_str):
            # 格式为[ccc, xxx, cid]的层级知识点，取最细粒度的作为concept
            c_ids = c_id_str.split(",")
            c_id = c_ids[-1]
            c_id = c_id.strip()
            return c_id

        df.dropna(subset=['Problem Name', 'Step Name', 'timestamp', 'correct'], inplace=True)
        df["concept_id"] = df["concept_id"].apply(process_concept_id)
        df["timestamp"] = df["timestamp"].apply(time_str2timestamp)
        df = df[df["correct"] != "hint"]
        df.loc[df["correct"] == "correct", "correct"] = 1
        df.loc[df["correct"] == "incorrect", "correct"] = 0

        df['Problem Name'] = df['Problem Name'].apply(replace_text)
        df['Step Name'] = df['Step Name'].apply(replace_text)
        df["question_id"] = df.apply(lambda x: "{}----{}".format(x["Problem Name"], x["Step Name"]), axis=1)
        df["question_id"] = df["question_id"].map(str)
        question_ids = pd.unique(df["question_id"])
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        self.question_id_map["single_concept"] = pd.DataFrame({
            "question_id": question_id_map.keys(),
            "question_mapped_id": question_id_map.values()
        })

        concept_ids = pd.unique(df["concept_id"])
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        df["concept_id"] = df["concept_id"].map(concept_id_map)
        self.concept_id_map["single_concept"] = pd.DataFrame({
            "concept_id": concept_id_map.keys(),
            "concept_mapped_id": concept_id_map.values()
        })

        df["user_id"] = df["user_id"].map(str)
        user_ids = pd.unique(df["user_id"])
        df["user_id"] = df["user_id"].map({u_id: i for i, u_id in enumerate(user_ids)})

        self.data_preprocessed["single_concept"] = df[["user_id", "timestamp", "concept_id", "correct", "question_id"]]
        self.statics_preprocessed["single_concept"] = DataProcessor.get_basic_info(self.data_preprocessed["single_concept"])

        # Q table
        df_new = pd.DataFrame({
            "question_id": map(int, df["question_id"].tolist()),
            "concept_id": map(int, df["concept_id"].tolist())
        })
        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
            correspond_c = pd.unique(group_info["concept_id"]).tolist()
            Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table["single_concept"] = Q_table

    def uniform_data(self):
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        if dataset_name == "assist2009":
            self.uniform_assist2009()
        elif dataset_name in ["assist2012", "assist2017"]:
            self.uniform_assist2012()
        elif dataset_name == "assist2015":
            self.uniform_assist2015()
        elif dataset_name in ["edi2020-task1", "edi2020-task34"]:
            self.uniform_edi2020()
        elif dataset_name == "xes3g5m":
            # 直接在load_process_uniform_xes3g5m里一起处理了
            pass
        elif dataset_name in ["assist2009-full", "ednet-kt1", "algebra2005", "algebra2006", "algebra2008", "bridge2algebra2006", "bridge2algebra2008"]:
            self.uniform_raw_is_single_concept()
        elif dataset_name in ["SLP-bio", "SLP-chi", "SLP-eng", "SLP-geo", "SLP-his", "SLP-mat", "SLP-phy"]:
            self.uniform_SLP()
        elif dataset_name == "statics2011":
            self.uniform_statics2011()
        else:
            raise NotImplementedError()

    def uniform_assist2009(self):
        df = deepcopy(self.data_preprocessed["multi_concept"])
        # school_id按照学生数量重映射
        df["school_id"] = df["school_id"].fillna(-1)
        df["school_id"] = df["school_id"].map(int)
        school_id_map, school_info = preprocess_raw.map_user_info(df, "school_id")
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        data_processed_dir = self.objects["file_manager"].get_preprocessed_dir(dataset_name)
        school_id_map_path = os.path.join(data_processed_dir, "school_id_map.csv")
        school_info_path = os.path.join(data_processed_dir, "school_info.json")
        school_id_map.to_csv(school_id_map_path, index=False)
        data_util.write_json(school_info, school_info_path)

        info_name_table = {
            "question_seq": "question_id",
            "concept_seq": "concept_id",
            "correct_seq": "correct"
        }

        # only_question
        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"order_id", "concept_name"})
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
        school_id_map_dict = school_id_map.to_dict()
        school_id_map4single_concept = {school_id_map_dict["school_id"][i]: school_id_map_dict["school_id_map"][i]
                                        for i in range(len(school_id_map_dict["school_id"]))}
        df["school_id"] = df["school_id"].map(school_id_map4single_concept)
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
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        data_processed_dir = self.objects["file_manager"].get_preprocessed_dir(dataset_name)
        school_id_map_path = os.path.join(data_processed_dir, "school_id_map.csv")
        school_info_path = os.path.join(data_processed_dir, "school_info.json")
        school_id_map.to_csv(school_id_map_path, index=False)
        data_util.write_json(school_info, school_info_path)

        info_name_table = {
            "question_seq": "question_id",
            "concept_seq": "concept_id",
            "correct_seq": "correct",
            "time_seq": "timestamp",
            "use_time_seq": "use_time"
        }

        # single_concept
        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"concept_name"})
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
        df = deepcopy(self.data_preprocessed["only_question"])
        info_name_table = {
            "question_seq": "question_id",
            "correct_seq": "correct"
        }
        id_keys = list(set(df.columns) - set(info_name_table.values()))
        dataset_seq_keys = CONSTANT.datasets_seq_keys()["assist2015"]
        seqs = []
        for user_id in pd.unique(df["user_id"]):
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["log_id"])
            object_data = {info_name: [] for info_name in dataset_seq_keys}
            for k in id_keys:
                object_data[k] = user_data.iloc[0][k]
            for i, (_, row_data) in enumerate(user_data.iterrows()):
                for info_name in dataset_seq_keys:
                    object_data[info_name].append(row_data[info_name_table[info_name]])
            object_data["seq_len"] = len(object_data["correct_seq"])
            seqs.append(object_data)
        self.data_uniformed["only_question"] = list(filter(lambda item: 2 <= item["seq_len"], seqs))

    def uniform_edi2020(self):
        df = deepcopy(self.data_preprocessed["single_concept"])
        info_name_table = {
            "question_seq": "question_id",
            "concept_seq": "concept_id",
            "correct_seq": "correct",
            "time_seq": "timestamp",
            "age_seq": "age"
        }
        id_keys = list(set(df.columns) - set(info_name_table.values()))
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        dataset_seq_keys = deepcopy(CONSTANT.datasets_seq_keys()[dataset_name])
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

    def uniform_SLP(self):
        df = deepcopy(self.data_preprocessed["single_concept"])
        # school_id按照学生数量重映射
        df["school_id"] = df["school_id"].fillna(-1)
        school_id_map, school_info = preprocess_raw.map_user_info(df, "school_id")
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        data_processed_dir = self.objects["file_manager"].get_preprocessed_dir(dataset_name)
        school_id_map_path = os.path.join(data_processed_dir, "school_id_map.csv")
        school_info_path = os.path.join(data_processed_dir, "school_info.json")
        school_id_map.to_csv(school_id_map_path, index=False)
        data_util.write_json(school_info, school_info_path)

        info_name_table = {
            "question_seq": "question_id",
            "concept_seq": "concept_id",
            "correct_seq": "correct",
            "interaction_type_seq": "interaction_type"
        }

        # single_concept
        id_keys = list(set(df.columns) - set(info_name_table.values()))
        dataset_seq_keys = CONSTANT.datasets_seq_keys()["SLP"]
        seqs = []
        for user_id in pd.unique(df["user_id"]):
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["timestamp", "order"])
            object_data = {info_name: [] for info_name in dataset_seq_keys}
            for k in id_keys:
                object_data[k] = user_data.iloc[0][k]
            for i, (_, row_data) in enumerate(user_data.iterrows()):
                for info_name in dataset_seq_keys:
                    object_data[info_name].append(row_data[info_name_table[info_name]])
            object_data["seq_len"] = len(object_data["correct_seq"])
            seqs.append(object_data)
        self.data_uniformed["single_concept"] = list(filter(lambda item: 2 <= item["seq_len"], seqs))

    def uniform_statics2011(self):
        df = deepcopy(self.data_preprocessed["single_concept"])
        info_name_table = {
            "question_seq": "question_id",
            "concept_seq": "concept_id",
            "correct_seq": "correct",
            "time_seq": "timestamp"
        }

        id_keys = list(set(df.columns) - set(info_name_table.values()))
        dataset_seq_keys = CONSTANT.datasets_seq_keys()["statics2011"]
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

    def process_raw_is_single_concept(self, df_single_concept, df_multi_concept, c_id_is_num=True):
        """
        处理像ednet-kt1、assist2009-full和kdd-cup这种数据，其每条记录是一道习题\n
        这类数据统一先提取single concept 和 multi concept的Q table，然后生成single concept数据，再根据single concept和Q table生成only question和multi concept\n
        这类数据由分为两类，一类是concept id为数字（ednet-kt1、assist2009-full、algebra2008、bridge2algebra2008）；
        另一类是concept id非数值（algebra2005、algebra2006、bridge2algebra2006）

        :param df_single_concept:
        :param df_multi_concept:
        :param c_id_is_num:
        :return:
        """
        # single_concept
        # 习题id重映射
        question_ids = list(pd.unique(df_single_concept["question_id"]))
        df_single_concept["question_id"] = df_single_concept["question_id"].map({q_id: i for i, q_id in enumerate(question_ids)})
        self.question_id_map["single_concept"] = pd.DataFrame({
            "question_id": question_ids,
            "question_id_map": range(len(question_ids))
        })

        # 知识点id重映射
        concept_ids = list(pd.unique(df_single_concept["concept_id"]))
        df_single_concept["concept_id"] = df_single_concept["concept_id"].map({c_id: i for i, c_id in enumerate(concept_ids)})
        self.concept_id_map["single_concept"] = pd.DataFrame({
            "concept_id": concept_ids,
            "concept_id_map": range(len(concept_ids))
        })

        df_new = pd.DataFrame({
            "question_id": map(int, df_single_concept["question_id"].tolist()),
            "concept_id": map(int, df_single_concept["concept_id"].tolist())
        })
        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
            correspond_c = pd.unique(group_info["concept_id"]).tolist()
            Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)

        self.data_preprocessed["single_concept"] = df_single_concept
        self.statics_preprocessed["single_concept"] = DataProcessor.get_basic_info(df_single_concept)
        self.statics_preprocessed["single_concept"]["num_concept"] = len(concept_ids)
        self.Q_table["single_concept"] = Q_table

        # multi_concept
        # df_multi_concept只是用来统计信息，知识点映射并不使用它，因为准备在uniform是直接用single2multi和q table转换为multi concept
        if c_id_is_num:
            self.process_c_id_num(df_multi_concept, df_new)
        else:
            self.process_c_id_not_num(df_multi_concept, df_new)

    def process_c_id_num(self, df_multi_concept, df_new):
        # 扩展为多知识点数据，即一条记录为一个知识点，而不是一道习题，类似skill_builder_data_corrected_collapsed.csv这种格式
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        split_table = {
            "ednet-kt1": "_",
            "assist2009-full": ";",
        }
        df_multi_concept["concept_id"] = df_multi_concept["concept_id"].str.split(split_table[dataset_name])
        df_multi_concept = df_multi_concept.explode("concept_id").reset_index(drop=True)
        self.statics_preprocessed["multi_concept"] = DataProcessor.get_basic_info(df_multi_concept)
        self.data_preprocessed["multi_concept"] = df_multi_concept

        question_ids = []
        concept_ids = []
        question_concept_map = {}
        concept_id_map = self.concept_id_map["single_concept"]
        for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
            correspond_c = pd.unique(group_info["concept_id"]).tolist()
            c_ids = concept_id_map.loc[concept_id_map["concept_id_map"] == correspond_c[0], "concept_id"].iloc[0]
            c_ids = list(map(int, c_ids.split(split_table[dataset_name])))
            question_ids.append(question_id)
            concept_ids.extend(c_ids)
            question_concept_map[question_id] = c_ids
        concept_ids = sorted(list(set(concept_ids)))
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        self.concept_id_map["multi_concept"] = pd.DataFrame({
            "concept_id": concept_id_map.keys(),
            "concept_id_map": concept_id_map.values()
        })
        for q_id in question_concept_map.keys():
            c_ids_map = []
            for c_id_ori in question_concept_map[q_id]:
                c_ids_map.append(concept_id_map[c_id_ori])
            question_concept_map[q_id] = c_ids_map
        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for question_id, correspond_c in question_concept_map.items():
            Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table["multi_concept"] = Q_table
        self.statics_preprocessed["multi_concept"]["num_max_concept"] = (int(max(Q_table.sum(axis=1))))

    def process_c_id_not_num(self, df_multi_concept, df_new):
        # 扩展为多知识点数据，即一条记录为一个知识点，而不是一道习题，类似skill_builder_data_corrected_collapsed.csv这种格式
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        split_table = {
            "algebra2005": "~~",
            "algebra2006": "~~",
            "algebra2008": "~~",
            "bridge2algebra2006": "~~",
            "bridge2algebra2008": "~~"
        }
        df_multi_concept["concept_id"] = df_multi_concept["concept_id"].str.split(split_table[dataset_name])
        df_multi_concept = df_multi_concept.explode("concept_id").reset_index(drop=True)
        self.statics_preprocessed["multi_concept"] = DataProcessor.get_basic_info(df_multi_concept)
        self.data_preprocessed["multi_concept"] = df_multi_concept

        question_ids = []
        concept_ids = []
        question_concept_map = {}
        concept_id_map = {}
        for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
            correspond_c = pd.unique(group_info["concept_id"]).tolist()
            # c_ids为原始数据的id，非数值
            c_ids = self.concept_id_map["single_concept"].loc[
                self.concept_id_map["single_concept"]["concept_id_map"] == correspond_c[0], "concept_id"
            ].iloc[0]
            c_ids = c_ids.split(split_table[dataset_name])
            # 这里就要转换为数值
            for c_id in c_ids:
                if c_id not in concept_id_map.keys():
                    concept_id_map[c_id] = len(concept_id_map)
            c_ids = list(map(lambda x: concept_id_map[x], c_ids))
            question_ids.append(question_id)
            concept_ids.extend(c_ids)
            question_concept_map[question_id] = c_ids
        concept_ids = list(set(concept_ids))
        self.concept_id_map["multi_concept"] = pd.DataFrame({
            "concept_id": concept_id_map.keys(),
            "concept_id_map": concept_id_map.values()
        })
        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for question_id, correspond_c in question_concept_map.items():
            Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table["multi_concept"] = Q_table
        self.statics_preprocessed["multi_concept"]["num_max_concept"] = (int(max(Q_table.sum(axis=1))))

    def uniform_raw_is_single_concept(self):
        dataset_name = self.params["preprocess_config"]["dataset_name"]
        df = deepcopy(self.data_preprocessed["single_concept"])
        info_name_table = {
            "question_seq": "question_id",
            "concept_seq": "concept_id",
            "correct_seq": "correct",
            "time_seq": "timestamp"
        }
        order_key_table = {
            "ednet-kt1": ["timestamp"],
            "assist2009-full": ["order_id"],
            "algebra2005": ["timestamp"],
            "algebra2006": ["timestamp"],
            "algebra2008": ["timestamp"],
            "bridge2algebra2006": ["timestamp"],
            "bridge2algebra2008": ["timestamp"],
        }
        if dataset_name == "ednet-kt1":
            info_name_table["use_time_seq"] = "use_time"
        if dataset_name in ["algebra2005", "algebra2006", "algebra2008", "bridge2algebra2006", "bridge2algebra2008"]:
            df["tmp_index"] = range(df.shape[0])
            order_key_table[dataset_name].append("tmp_index")

        id_keys = list(set(df.columns) - set(info_name_table.values()))
        dataset_seq_keys = CONSTANT.datasets_seq_keys()[dataset_name]
        seqs = []
        for ui, user_id in enumerate(pd.unique(df["user_id"])):
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=order_key_table[dataset_name])
            if dataset_name == "ednet-kt1":
                user_data["timestamp"] = user_data["timestamp"].map(lambda x: int(x / 1000))
            object_data = {info_name: [] for info_name in dataset_seq_keys}
            for k in id_keys:
                object_data[k] = user_data.iloc[0][k]
            if type(object_data["user_id"]) is not int:
                object_data["user_id"] = ui
            if "tmp_index" in object_data.keys():
                del object_data["tmp_index"]
            for i, (_, row_data) in enumerate(user_data.iterrows()):
                for info_name in dataset_seq_keys:
                    object_data[info_name].append(int(row_data[info_name_table[info_name]]))
            object_data["seq_len"] = len(object_data["correct_seq"])
            seqs.append(object_data)

        self.data_uniformed["single_concept"] = list(filter(lambda item: 2 <= item["seq_len"], seqs))

        data_only_question = deepcopy(seqs)
        for item_data in data_only_question:
            del item_data["concept_seq"]
        self.data_uniformed["only_question"] = list(filter(lambda item: 2 <= item["seq_len"], data_only_question))

        seqs = DataProcessor.single2multi(data_only_question, self.Q_table["multi_concept"])
        self.data_uniformed["multi_concept"] = list(filter(lambda item: 2 <= item["seq_len"], seqs))

    def get_all_id_maps(self):
        result = {}
        for data_type, id_map in self.question_id_map.items():
            result.setdefault(data_type, {})
            result[data_type]["question_id_map"] = id_map

        for data_type, id_map in self.concept_id_map.items():
            result.setdefault(data_type, {})
            result[data_type]["concept_id_map"] = id_map

        return result

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
        id_keys, seq_keys = parse_util.get_keys_from_uniform(seqs)
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
                c_ids = parse_util.get_concept_from_question(q_id, Q_table)
                len_c_ids = len(c_ids)
                item_data_new["question_seq"] += [q_id] + [-1] * (len_c_ids - 1)
                item_data_new["concept_seq"] += c_ids
                for i, info_name in enumerate(seq_keys):
                    item_data_new[info_name] += [ele_all[i + 1]] * len_c_ids
            item_data_new["seq_len"] = len(item_data_new["correct_seq"])
            seqs_new.append(item_data_new)
        return seqs_new
