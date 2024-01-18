import pandas as pd
import numpy as np

from copy import deepcopy

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

        return {
            "data_processed": df,
            "Q_table": Q_table,
            "concept_id_map": pd.DataFrame({
                "concept_id": concept_id_map.keys(),
                "concept_mapped_id": concept_id_map.values()
            }),
            "question_id_map": pd.DataFrame({
                "question_id": question_id_map.keys(),
                "question_mapped_id": question_id_map.values()
            })
        }

    else:
        question_ids = pd.unique(df["question_id"])
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        return {
            "data_processed": df,
            "question_id_map": pd.DataFrame({
                "question_id": question_id_map.keys(),
                "question_mapped_id": question_id_map.values()
            })
        }


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
    if dataset_name in ["assist2009", "assist2009-full"]:
        df = multi_concept2single_concept4assist2009(df)
    result["single_concept"] = qc_id_remap(dataset_name, df)

    return result


def map_user_info(df, field):
    # 将用户的指定信息进行重映射，并按照用户数量排序（如重映射学校id，那么学生数量大的学校id被映射为0，其它学校依次映射为1,2···）
    num_user_in_field = df[df[field] != -1].groupby(field).agg(user_count=("user_id", lambda x: x.nunique())).to_dict()
    num_user_in_field = list(num_user_in_field["user_count"].items())
    num_user_in_field = sorted(num_user_in_field, key=lambda item: item[1], reverse=True)
    field_id_map = {item[0]: i for i, item in enumerate(num_user_in_field)}
    field_id_map[-1] = -1
    df[field] = df[field].map(field_id_map)

    num_user_in_field = list(map(lambda item: (field_id_map[item[0]], item[1]), num_user_in_field))
    field_id_map = pd.DataFrame({
        field: list(field_id_map.keys()),
        f"{field}_map": list(field_id_map.values())
    })
    field_info = {field_id: {
        "num_user": num_user,
        "num_interaction": len(df[df[field] == field_id])
    } for field_id, num_user in num_user_in_field}
    field_info[-1] = {
        "num_user": len(pd.unique(df[df[field] == -1]["user_id"])),
        "num_interaction": len(df[df[field] == -1])
    }

    return field_id_map, field_info
