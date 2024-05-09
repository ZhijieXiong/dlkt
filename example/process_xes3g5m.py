import json
import os
import re
import numpy as np


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
        contents.append({"contentType": "TEXT", "chinese": question_text, "english": ""})
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
    its_question = {}
    error_question = {}
    for q_id, q_content in QUESTION_META_DATA.items():
        question_text = q_content["content"]
        question_images = QUESTION_IMAGE[q_id]
        analysis_text = q_content["analysis"]
        analysis_images = ANALYSIS_IMAGE[q_id]

        try:
            question_contents = text2contents(question_text, question_images)
            analysis_contents = text2contents(analysis_text, analysis_images)
            its_question[q_id] = {
                "question_contents": question_contents,
                "analysis_contents": analysis_contents,
                "question_type": q_content["type"],
                "concept_routes": q_content["kc_routes"],
                "options": q_content["options"] if q_content["type"] == "单选" else dict(),
                "answer": list(map(lambda x: x.strip("$"), q_content["answer"]))
            }
        except IndexError:
            # 报错的单独手动处理
            error_question[q_id] = {
                "question_text": question_text,
                "analysis_text": analysis_text,
                "question_images": question_images,
                "analysis_images": analysis_images,
                "question_type": q_content["type"],
                "concept_routes": q_content["kc_routes"],
                "options": q_content["options"] if q_content["type"] == "单选" else dict(),
                "answer": list(map(lambda x: x.strip("$"), q_content["answer"]))
            }
    write_json(its_question, os.path.join(OUTPUT_DIR, "its_question.json"))
    write_json(error_question, os.path.join(OUTPUT_DIR, "error_question.json"))
