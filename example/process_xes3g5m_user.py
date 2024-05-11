import os

import config

from lib.util.data import read_preprocessed_file, generate_unique_id, load_json, write_json


if __name__ == "__main__":
    data_path = r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\xes3g5m\data_only_question.txt"
    its_question_path = r"F:\code\myProjects\dlkt\lab\math_dataset\xes3g5m\its_question.json"
    user_data_dir = r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\xes3g5m"
    data = read_preprocessed_file(data_path)
    its_question = load_json(its_question_path)

    its_user_behavior = []
    its_user = []
    user_ids = []
    for item_data in data:
        user_id = item_data["user_id"]
        while user_id in user_ids:
            user_id += 1
        its_user.append({
            "name": generate_unique_id(f"xes3g5m-user-data-{user_id}"),
            "password": f"xes3g5m{user_id}",
            "email": f"xes3g5m{user_id}@163.com"

        })
        for i in range(item_data["seq_len"]):
            its_user_behavior.append({
                "user_name": generate_unique_id(f"xes3g5m-user-data-{user_id}"),
                "question_id": its_question[str(item_data["question_seq"][i])]["question_id"],
                "correct": bool(item_data["correct_seq"][i]),
                "timestamp": item_data["time_seq"][i]
            })
        user_ids.append(user_id)

    write_json(its_user, os.path.join(user_data_dir, "its_user.json"))
    write_json(its_user_behavior, os.path.join(user_data_dir, "its_user_behavior.json"))
