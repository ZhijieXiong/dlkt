import os
import pandas as pd


def load_csv(data_path, useful_cols=None, rename_dict=None, num_rows=None):
    try:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="utf-8", low_memory=False, index_col=False, nrows=num_rows)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, index_col=False, nrows=num_rows)
    if rename_dict is not None:
        df.rename(columns=rename_dict, inplace=True)
    return df


def load_table(data_path, useful_cols=None, rename_dict=None, num_rows=None):
    try:
        df = pd.read_table(data_path, usecols=useful_cols, encoding="utf-8", low_memory=False, nrows=num_rows)
    except UnicodeDecodeError:
        df = pd.read_table(data_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, nrows=num_rows)
    if rename_dict is not None:
        df.rename(columns=rename_dict, inplace=True)
    return df


def load_SLP(data_dir, dataset_name):
    subject = dataset_name.split("-")[-1]
    unit_path = os.path.join(data_dir, f"unit-{subject}.csv")
    term_path = os.path.join(data_dir, f"term-{subject}.csv")
    student_path = os.path.join(data_dir, "student.csv")
    family_path = os.path.join(data_dir, "family.csv")
    # school_path = os.path.join(data_dir, "school.csv")

    useful_cols = ["student_id", "question_id", "concept", "score", "full_score", "time_access"]
    family_cols = ["student_id", "live_on_campus"]
    student_cols = ["student_id", "gender", "school_id"]
    # school_cols = ["school_id", "school_type"]

    unit = load_csv(unit_path, useful_cols)
    term = load_csv(term_path, useful_cols)
    student = load_csv(student_path, student_cols)
    family = load_csv(family_path, family_cols)
    # school = load_csv(school_path, school_cols)

    # 原文件已经是排过序的，加上order方便后面利用
    unit["order"] = range(len(unit))
    term["order"] = range(len(unit), len(unit) + len(term))
    # 将总评数据加入
    student_ids = pd.unique(unit["student_id"])
    student_df = pd.DataFrame({"student_id": student_ids})

    # unit为0，term为1
    unit.insert(loc=len(unit.columns), column='interaction_type', value=0)
    term = student_df.merge(term, how="left", on=["student_id"])
    term.insert(loc=len(term.columns), column='interaction_type', value=1)
    df = pd.concat([unit, term], axis=0)

    df = df.merge(family, how="left", on=["student_id"])
    df = df.merge(student, how="left", on=["student_id"])
    # df = df.merge(school, how="left", on=["school_id"])

    # live_on_campus和school_type有nan
    return df[["student_id", "question_id", "concept", "score", "full_score", "time_access", "order",
               "live_on_campus", "school_id", "gender", "interaction_type"]]


def load_ednet_kt1(data_dir, num_file=1):
    # 多知识点算新知识点
    dfs = []

    def process_tags(tags_str):
        # 多知识点是用_连接的，但是如 1_2_3 和 2_3_1 表示同一多知识点组合，所以统一表示成id小的在前面，即1_2_3
        tags = tags_str.split("_")
        tags = list(map(str, sorted(list(map(int, tags)))))
        return "_".join(tags)

    for i in range(num_file):
        file_name = f"users_{i}.csv"
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            break
        else:
            try:
                df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
            df["tags"] = df["tags"].map(process_tags)
            dfs.append(df)

    return pd.concat(dfs, axis=0)


