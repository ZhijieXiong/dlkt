import json
import time

import config

from lib.util.data import load_json, write2file


def time_str2timestamp(time_str):
    # if len(time_str) != 19:
    #     time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
    return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    data_path = "/Users/dream/myProjects/dlkt-release/example/lab/dataset_raw/moocradar/student-problem-coarse.json"
    problem_data_path = "/Users/dream/myProjects/dlkt-release/example/lab/dataset_raw/moocradar/problem.json"

    # data = load_json(data_path)
    problem_data = []
    with open(problem_data_path, "r") as f:
        for line in f:
            problem_data.append(json.loads(line))

    num_question = len(problem_data)
    q_id_map = {}
    concepts = set()
    courses = set()
    courses_example = {}
    question_meta_data = {}
    # 只有2个错误
    num_err_eval = 0
    for i, p in enumerate(problem_data):
        q_id_map[p["problem_id"]] = i
        concepts.update(p["concepts"])
        courses.add(p["course_id"])
        if p["course_id"] not in courses_example:
            courses_example[p["course_id"]] = i

        try:
            p_detail = eval(p["detail"])
            question_meta_data[i] = {
                "original_id": p["problem_id"],
                "course_id": p["course_id"],
                "type": p_detail["typetext"],
                "content": p_detail["content"],
                "option": p_detail["option"],
                "answer": p_detail["answer"],
                "concept_ids": p["concepts"]
            }
        except:
            print(p["problem_id"])
            num_err_eval += 1
            question_meta_data[i] = {
                "original_id": p["problem_id"],
                "course_id": p["course_id"],
                "concept_ids": p["concepts"]
            }

    # 手动处理错误eval
    question_meta_data[q_id_map["Pm_1308411"]].update({
        "type": "判断题",
        "content": "政府购买与政府支出没有区别",
        "option": None,
        "answer": ["false"],
    })
    question_meta_data[q_id_map["Pm_8045638"]].update({
        "type": "单选题",
        "content": "以下哪一项不属于创业者必须具备的能力:()",
        "option": {'A': '人际协调能力', 'B': '开拓创新能力', 'C': '组织管理能力', 'D': '自我执行能力'},
        "answer": ["D"],
    })

    # 看一下有哪些课程
    print(f"There are {len(courses)} courses")
    for course_id in courses_example:
        print(course_id)
        print(question_meta_data[courses_example[course_id]]["content"])

    num_concept = len(concepts)
    c_id_map = {c_name: i for i, c_name in enumerate(concepts)}
    for q_id in question_meta_data:
        question_meta_data[q_id]["concept_ids"] = (
            list(map(lambda c_name: c_id_map[c_name], question_meta_data[q_id]["concept_ids"]))
        )

    data_only_question = []

