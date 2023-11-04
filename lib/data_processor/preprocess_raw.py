import pandas as pd
import numpy as np

from copy import deepcopy

from . import CONSTANT
from .. import DATASET_INFO


def qc_id_remap(dataset_name, df):
    # 多知识点拆分为新知识点
    df = deepcopy(df)
    if dataset_name in DATASET_INFO.datasets_has_q_table():
        concept_ids = pd.unique(df["concept_id"])
        question_ids = pd.unique(df["question_id"])
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        df["concept_id"] = df["concept_id"].map(concept_id_map)

        df_new = pd.DataFrame({
            "question_id": map(int, df["question_id"].tolist()),
            "concept_id": map(int, df["concept_id"].tolist())
        })
        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
            correspond_c = pd.unique(group_info["concept_id"]).tolist()
            Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)

        return {"data_processed": df, "Q_table": Q_table,
                "concept_id_map": concept_id_map, "question_id_map": question_id_map}
    else:
        question_ids = pd.unique(df["question_id"])
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        return {"data_processed": df, "question_id_map": question_id_map}


def multi_concept2single_concept4assist2009(df):
    # 多知识点算新知识点，输入的df中concept_id列格式为0_1，表示这道习题对应知识点0和1 ，并且知识点要排好序，如1_0要处理成0_1
    # 多知识点数据集经过这个处理以后，变成单知识点数据集
    df = deepcopy(df)
    df_new = pd.DataFrame({
        "question_id": map(int, df["question_id"].tolist()),
        "concept_id": map(int, df["concept_id"].tolist())
    })
    q_c_table = {}
    for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
        # 假设question_id为0，对应知识点为1、2
        # c_str就是将习题0对应的知识点设为"1_2"
        c_str = "_".join(list(map(str, sorted(pd.unique(group_info["concept_id"]).tolist()))))
        q_c_table.setdefault(question_id, c_str)
    # 去除多知识点习题的冗余
    df = df[~df.duplicated(subset=["user_id", "order_id", "question_id"])]
    df["concept_id"] = df["question_id"].map(lambda q_id: q_c_table[q_id])
    return df


def preprocess_assist(dataset_name, df):
    """
    需要返回的值：预处理后的数据、Q table、id映射
    预处理：multi concept、single concept
    id映射：习题、知识点
    """
    result = {
        "multi_concept": None,
        "single_concept": None
    }

    if dataset_name in DATASET_INFO.datasets_multi_concept():
        result["multi_concept"] = qc_id_remap(dataset_name, df)
    if dataset_name in ["assist2009", "assist2009-new"]:
        df = multi_concept2single_concept4assist2009(df)
    result["single_concept"] = qc_id_remap(dataset_name, df)

    return result
