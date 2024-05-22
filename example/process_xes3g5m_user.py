import os

import config

from lib.util.data import read_preprocessed_file, load_json, write_json


if __name__ == "__main__":
    data_path = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition\test_data.txt"
    its_question_path = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition\its_question.json"
    user_data_dir = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition"
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
            "name": f"xes3g5mUser{user_id}",
            "password": f"xes3g5mUser{user_id}",
            "email": f"xes3g5mUser{user_id}@163.com"
        })
        for i in range(item_data["seq_len"]):
            its_user_behavior.append({
                "user_name": f"xes3g5mUser{user_id}",
                "question_id": its_question[str(item_data["question_seq"][i])]["question_id"],
                "question_type": its_question[str(item_data["question_seq"][i])]["question_type"],
                "correct": bool(item_data["correct_seq"][i]),
                "timestamp": item_data["time_seq"][i] // 1000
            })
        user_ids.append(user_id)

    write_json(its_user, os.path.join(user_data_dir, "its_user.json"))
    write_json(its_user_behavior, os.path.join(user_data_dir, "its_user_behavior.json"))
