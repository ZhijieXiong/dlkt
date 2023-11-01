import os
import pandas as pd


def load_csv(data_path, useful_cols=None, rename_dict=None):
    try:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False)
    if rename_dict is not None:
        df.rename(columns=rename_dict, inplace=True)
    return df


def load_SLP(data_dir, dataset_name):
    subject = dataset_name.split("-")[-1]
    unit_path = os.path.join(data_dir, f"unit-{subject}.csv")
    term_path = os.path.join(data_dir, f"term-{subject}.csv")
    student_path = os.path.join(data_dir, "student.csv")
    family_path = os.path.join(data_dir, "family.csv")
    school_path = os.path.join(data_dir, "school.csv")

    useful_cols = ["student_id", "question_id", "concept", "score", "full_score", "time_access"]
    family_cols = ["student_id", "live_on_campus"]
    student_cols = ["student_id", "gender", "school_id"]
    school_cols = ["school_id", "school_type"]

    unit = load_csv(unit_path, useful_cols)
    term = load_csv(term_path, useful_cols)
    student = load_csv(student_path, student_cols)
    family = load_csv(family_path, family_cols)
    school = load_csv(school_path, school_cols)

    # 原文件已经是排过序的，加上order方便后面利用
    unit["order"] = range(len(unit))
    term["order"] = range(len(unit), len(unit) + len(term))
    # 将总评数据加入
    student_ids = pd.unique(unit["student_id"])
    student_df = pd.DataFrame({"student_id": student_ids})
    df = pd.concat([unit, student_df.merge(term, how="left", on=["student_id"])], axis=0)

    df = df.merge(family, how="left", on=["student_id"])
    df = df.merge(student, how="left", on=["student_id"])
    df = df.merge(school, how="left", on=["school_id"])

    # live_on_campus和school_type有nan
    return df[["student_id", "question_id", "concept", "score", "full_score", "time_access", "order",
               "live_on_campus", "school_type", "gender"]]


def load_ednet_kt1(data_dir):
    # 多知识点算新知识点
    dfs = []

    def process_tags(tags_str):
        tags = tags_str.split("_")
        tags = list(map(str, sorted(list(map(int, tags)))))
        return "_".join(tags)

    for i in range(200):
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

