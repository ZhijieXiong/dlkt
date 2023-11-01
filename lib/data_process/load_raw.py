import os
import pandas as pd


def load_csv(data_path, useful):
    try:
        df = pd.read_csv(data_path, usecols=useful, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, usecols=useful, encoding="ISO-8859-1", low_memory=False)
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

    try:
        unit = pd.read_csv(unit_path, usecols=useful_cols, encoding="utf-8", low_memory=False, index_col=False)
    except UnicodeDecodeError:
        unit = pd.read_csv(unit_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, index_col=False)
    try:
        term = pd.read_csv(term_path, usecols=useful_cols, encoding="utf-8", low_memory=False, index_col=False)
    except UnicodeDecodeError:
        term = pd.read_csv(term_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, index_col=False)
    try:
        student = pd.read_csv(student_path, usecols=student_cols, encoding="utf-8", low_memory=False, index_col=False)
    except UnicodeDecodeError:
        student = pd.read_csv(student_path, usecols=student_cols, encoding="ISO-8859-1", low_memory=False, index_col=False)
    try:
        family = pd.read_csv(family_path, usecols=family_cols, encoding="utf-8", low_memory=False, index_col=False)
    except UnicodeDecodeError:
        family = pd.read_csv(family_path, usecols=family_cols, encoding="ISO-8859-1", low_memory=False, index_col=False)
    try:
        school = pd.read_csv(school_path, usecols=school_cols, encoding="utf-8", low_memory=False, index_col=False)
    except UnicodeDecodeError:
        school = pd.read_csv(school_path, usecols=school_cols, encoding="ISO-8859-1", low_memory=False, index_col=False)

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


