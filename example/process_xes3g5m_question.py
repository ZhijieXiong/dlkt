import json
import os
import re
import hashlib
import numpy as np

from collections import defaultdict


def generate_unique_id(input_str):
    hash_object = hashlib.md5(input_str.encode())
    return hash_object.hexdigest()


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def write_json(json_data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def text2contents(text, images):
    contents = []

    if len(images) == 0:
        contents.append({"contentType": "TEXT", "chinese": text, "english": ""})
    else:
        second_text = text
        for image in images:
            two_text = second_text.split(image)
            first_text = two_text[0].strip()
            second_text = two_text[1]

            contents.append({"contentType": "TEXT", "chinese": first_text, "english": ""})
            contents.append({"contentType": "IMAGE", "imageName": image})
        second_text = second_text.strip("")
        if second_text != "":
            contents.append({"contentType": "TEXT", "chinese": second_text, "english": ""})

    return contents


def question2concept_from_Q(Q_table):
    # 将Q table转换为{question_id1: [concept_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[0])}
    return result


if __name__ == "__main__":
    # 原始数据文件的地址
    DATA_DIR = r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\xes3g5m"
    IMAGE_DIR = r"F:\code\myProjects\dlkt\lab\dataset_raw\xes3g5m\metadata\images"
    OUTPUT_DIR = r"F:\code\myProjects\dlkt\lab\math_dataset\xes3g5m"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    Q_TABLE = np.load(os.path.join(DATA_DIR, "Q_table_multi_concept.npy"))
    QUESTION_META_DATA = load_json(os.path.join(DATA_DIR, "question_meta.json"))
    CONCEPT_META_DATA = load_json(os.path.join(DATA_DIR, "concept_meta.json"))

    question_image_path = os.path.join(OUTPUT_DIR, "question_image.json")
    QUESTION_IMAGE = {}
    if os.path.exists(question_image_path):
        QUESTION_IMAGE = load_json(question_image_path)
    else:
        for k, v in QUESTION_META_DATA.items():
            QUESTION_IMAGE[k] = re.findall(r"question_\d+-image_\d+", v["content"])
        write_json(QUESTION_IMAGE, question_image_path)

    analysis_image_path = os.path.join(OUTPUT_DIR, "analysis_image.json")
    ANALYSIS_IMAGE = {}
    if os.path.exists(analysis_image_path):
        ANALYSIS_IMAGE = load_json(analysis_image_path)
    else:
        for k, v in QUESTION_META_DATA.items():
            ANALYSIS_IMAGE[k] = re.findall(r"analysis_\d+-image_\d+", v["analysis"])
        write_json(ANALYSIS_IMAGE, analysis_image_path)

    question_content_path = os.path.join(OUTPUT_DIR, "question_content.json")
    QUESTION_CONTENT = {}
    if os.path.exists(question_content_path):
        QUESTION_CONTENT = load_json(question_content_path)
    else:
        for k, v in QUESTION_META_DATA.items():
            QUESTION_CONTENT[k] = {
                "content_only_text": v["content"],
                "content_with_image_name": v["content"],
                "analysis_only_text": v["analysis"],
                "analysis_with_image_name": v["analysis"],
                "content_images": QUESTION_IMAGE[k],
                "analysis_images": ANALYSIS_IMAGE[k],
                "type": v["type"],
                "answer": v["answer"],
                "kc_routes": v["kc_routes"]
            }
            for image_str in QUESTION_IMAGE[k]:
                QUESTION_CONTENT[k]["content_only_text"] = QUESTION_CONTENT[k]["content_only_text"].replace(image_str, "")
                QUESTION_CONTENT[k]["content_with_image_name"] = QUESTION_CONTENT[k]["content_with_image_name"].replace(
                    image_str, f"[%{image_str}%]")
            QUESTION_CONTENT[k]["content_only_text"] = QUESTION_CONTENT[k]["content_only_text"].strip()
            QUESTION_CONTENT[k]["content_with_image_name"] = QUESTION_CONTENT[k]["content_with_image_name"].strip()

            for image_str in ANALYSIS_IMAGE[k]:
                QUESTION_CONTENT[k]["analysis_only_text"] = QUESTION_CONTENT[k]["analysis_only_text"].replace(image_str, "")
                QUESTION_CONTENT[k]["analysis_with_image_name"] = QUESTION_CONTENT[k]["analysis_with_image_name"].replace(
                    image_str, f"[%{image_str}%]")

            QUESTION_CONTENT[k]["analysis_only_text"] = QUESTION_CONTENT[k]["analysis_only_text"].strip()
            QUESTION_CONTENT[k]["analysis_with_image_name"] = QUESTION_CONTENT[k]["analysis_with_image_name"].strip()
        write_json(QUESTION_CONTENT, question_content_path)

    # 处理为最终格式，用于ITS系统
    OUTPUT_DIR = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition"
    # KT的习题id和ITS的习题id互相转换
    kt_id2its_id = {}
    its_id2kt_id = {}
    its_question_path = os.path.join(OUTPUT_DIR, "its_question.json")
    error_question_path = os.path.join(OUTPUT_DIR, "error_question.json")
    rag_lib_path = os.path.join(OUTPUT_DIR, "its_rag_question_lib.txt")
    rag_question_lib = []
    if os.path.exists(its_question_path) and os.path.exists(error_question_path) and os.path.exists(rag_lib_path):
        its_question = load_json(its_question_path)
        error_question = load_json(error_question_path)
    else:
        its_question = {}
        error_question = {}
        for q_id, q_content in QUESTION_META_DATA.items():
            question_text = q_content["content"]
            question_images = QUESTION_IMAGE[q_id]
            analysis_text = q_content["analysis"]
            analysis_images = ANALYSIS_IMAGE[q_id]
            its_question_id = generate_unique_id(f"xes3g5m-question-data-{q_id}")
            concept_kc_routes = list(filter(lambda x: "知识点" in x[:6], q_content["kc_routes"]))
            concept_kc_route_str = "；".join(concept_kc_routes)
            last_in_kc_routes = list(set(map(lambda x: x.split("----")[-1].strip(), q_content["kc_routes"])))
            lib_text = f"习题id：{its_question_id}\n" \
                       f"习题考察的知识点: {concept_kc_route_str}\n" \
                       f"习题关联的知识点或能力：{'， '.join(last_in_kc_routes)}\n" \
                       f"习题文本：{question_text}\n" \
                       f"习题答案与解释：{analysis_text}"
            rag_question_lib.append(lib_text)
            try:
                question_contents = text2contents(question_text, question_images)
                analysis_contents = text2contents(analysis_text, analysis_images)
                its_question[q_id] = {
                    "question_id": its_question_id,
                    "question_contents": question_contents,
                    "analysis_contents": analysis_contents,
                    "question_type": q_content["type"],
                    "concept_routes": q_content["kc_routes"],
                    "options": q_content["options"] if q_content["type"] == "单选" else dict(),
                    "answer": list(map(lambda x: x.strip("$"), q_content["answer"]))
                }
                kt_id2its_id[q_id] = generate_unique_id(f"xes3g5m-question-data-{q_id}")
                its_id2kt_id[generate_unique_id(f"xes3g5m-question-data-{q_id}")] = q_id
            except IndexError:
                # 报错的单独手动处理(只有一道题有问题，是该tag下第23题，填空题)
                error_question[q_id] = {
                    "question_id": its_question_id,
                    "question_text": question_text,
                    "analysis_text": analysis_text,
                    "question_images": question_images,
                    "analysis_images": analysis_images,
                    "question_type": q_content["type"],
                    "concept_routes": q_content["kc_routes"],
                    "options": q_content["options"] if q_content["type"] == "单选" else dict(),
                    "answer": list(map(lambda x: x.strip("$"), q_content["answer"]))
                }
        with open(rag_lib_path, "w", encoding="utf-8") as f:
            lines = "\n##############################\n".join(rag_question_lib)
            f.write(lines)
        write_json(its_question, os.path.join(OUTPUT_DIR, "its_question.json"))
        write_json(error_question, os.path.join(OUTPUT_DIR, "error_question.json"))
        write_json(kt_id2its_id, os.path.join(OUTPUT_DIR, "kt_id2its_id.json"))
        write_json(its_id2kt_id, os.path.join(OUTPUT_DIR, "its_id2kt_id.json"))

    # 创建符合ITS-back要求格式的数据
    q2c = question2concept_from_Q(np.load(os.path.join(DATA_DIR, "Q_table_multi_concept.npy")))
    its_single_choice_question = {
        "source": "xes3g5m",
        "link": "",
        "subjectType": "MATH",
        "exercises": []
    }
    its_fill_up_question = {
        "source": "xes3g5m",
        "link": "",
        "subjectType": "MATH",
        "exercises": []
    }
    tags_count = defaultdict(int)
    options_keys = list(map(lambda x: chr(ord("A") + x), list(range(26))))
    for q_id, question in its_question.items():
        question_type = question["question_type"]

        # 直接获取的内容
        concepts = question["concept_routes"]
        question_id = question["question_id"]
        exercise_contents = question["question_contents"]
        explanation_contents = question["analysis_contents"]
        correct_answer = question["answer"]

        # 获取tags和orderInTag
        tags = list(map(lambda x: x.split("----")[-1], concepts))
        c_ids = sorted(q2c[int(q_id)])
        c_ids = "----".join(list(map(str, c_ids)))
        order_in_tag = tags_count[c_ids]
        tags_count[c_ids] += 1

        if question_type != "单选":
            # 填空
            its_fill_up_question["exercises"].append({
                "orderInTag": order_in_tag,
                "id": question_id,
                "concepts": concepts,
                "tags": tags,
                "exerciseContents": exercise_contents,
                "explanationContents": explanation_contents,
                "correctAnswer": correct_answer
            })
        else:
            if len(correct_answer) > 1:
                # 一道选择题多个问题，目前不处理
                continue
            correct_answer = ord(correct_answer[0].upper()) - ord('A')
            # 获取options
            options = []
            for option_key in options_keys:
                if option_key in question["options"].keys():
                    options.append([{
                        "contentType": "TEXT",
                        "english": "",
                        "chinese": question["options"][option_key]
                    }])
            its_single_choice_question["exercises"].append({
                "orderInTag": order_in_tag,
                "id": question_id,
                "concepts": concepts,
                "tags": tags,
                "exerciseContents": exercise_contents,
                "options": options,
                "explanationContents": explanation_contents,
                "correctAnswer": correct_answer
            })
    write_json(its_single_choice_question, os.path.join(OUTPUT_DIR, "its_single_choice_question.json"))
    write_json(its_fill_up_question, os.path.join(OUTPUT_DIR, "its_fill_up_question.json"))
