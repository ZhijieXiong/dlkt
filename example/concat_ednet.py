import argparse
import os
import random
import pandas as pd

import config

from lib.util.FileManager import FileManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_manager_root", type=str, default=r"")
    parser.add_argument("--dataset_src_dir", type=str, default=r"E:\dataSet\knowledgeTracingtData\EDnet\EdNet-KT1\KT1")
    parser.add_argument("--contents_dir", type=str, default=r"E:\dataSet\knowledgeTracingtData\EDnet\EdNet-Contents\contents")
    args = parser.parse_args()
    params = vars(args)

    if params["file_manager_root"] == "":
        params["file_manager_root"] = config.FILE_MANAGER_ROOT

    file_manager = FileManager(params["file_manager_root"])

    data_dir = params["dataset_src_dir"]
    content_dir = params["contents_dir"]
    question_content_path = os.path.join(content_dir, "questions.csv")
    save_path = os.path.join(file_manager.get_root_dir(), "lab/dataset_raw/ednet-kt1")
    question_content = pd.read_csv(question_content_path, usecols=["question_id", "correct_answer", "tags"])
    question_content['tags'] = question_content['tags'].apply(lambda x: x.replace(";", "_"))
    question_content = question_content[question_content['tags'] != '-1']
    user_ids = [i for i in range(782000)]
    random.shuffle(user_ids)

    count = 0
    users5000_count = 0
    files = []
    for uid in user_ids:
        str_unum = str(uid)
        df_path = os.path.join(data_dir, f"./u{uid}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            df['user_id'] = uid
            files.append(df)
            count = count + 1
            if (count % 5000) != 0:
                continue
            df_save = pd.concat(files, axis=0, ignore_index=True)
            df_save = pd.merge(df_save, question_content, how="left", on="question_id")
            df_save = df_save.dropna(subset=["user_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
            df_save['correct'] = (df_save['correct_answer'] == df_save['user_answer']).apply(int)
            df_save = df_save[["timestamp", "question_id", "elapsed_time", "user_id", "tags", "correct"]]
            df_save.to_csv(os.path.join(save_path, f"./users_{users5000_count}.csv"), index=False)
            files = []
            users5000_count += 1
