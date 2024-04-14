import re
import os
import inspect
import json

from lib.util.data import load_json, write_json
from lib.api.llm_api import *

# 导入key
with open(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), "../keys.json"), "r") as f:
    KEYS = json.load(f)
OPENAI_KEY = KEYS["openai_key_from_lwd2hzhp"]

# 导入prompt template相关内容，如profile、example
with open(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), "../prompt_template/llm_profiles.json"), "r") as f:
    LLM_PROFILES = json.load(f)
with open(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), "../prompt_template/few_shot_examples.json"), "r") as f:
    FEW_SHOT_EXAMPLES = json.load(f)


def translate_question2english(model_name, question_type, question_text):
    if question_type == "multi_choice":
        profile_message = LLM_PROFILES["translator_xes3g5m-multi-choice-question_chinese2english"]
        translation_examples = FEW_SHOT_EXAMPLES["translator_xes3g5m-multi-choice-question_chinese2english"]
    elif question_type == "fill_up":
        profile_message = LLM_PROFILES["translator_xes3g5m-fill-up-question_chinese2english"]
        translation_examples = FEW_SHOT_EXAMPLES["translator_xes3g5m-fill-up-question_chinese2english"]
    else:
        raise NotImplementedError()
    answer = zero_or_few_shot(OPENAI_KEY, model_name, profile_message, translation_examples, question_text, "fake")
    return answer


if __name__ == "__main__":
    # 调用api的一些参数，比如"gpt-3.5-turbo", "gpt-4"
    MODEL_NAME = "gpt-4"
    NUM_CALL_API = 30

    # 原始数据文件的地址
    DATA_DIR = r"/Users/dream/myProjects/dlkt/lab/dataset_preprocessed/xes3g5m"
    IMAGE_DIR = r"/Users/dream/myProjects/dlkt/lab/dataset_raw/xes3g5m/metadata/images"
    QUESTION_META_DATA = load_json(f"{DATA_DIR}/question_meta.json")

    # 保存输出的地址（定了以后不要变更，为了稳定以及节约成本，该脚本会运行多次，如每次翻译n道题，所以需要读取之前已做过的操作）
    OUTPUT_DIR = r"/Users/dream/myProjects/dlkt/lab/math_dataset/xes3g5m"

    # 对数据进行一些预处理（生成三个文件：question_image.json, analysis_image, question_content）
    question_image_path = os.path.join(OUTPUT_DIR, "question_image.json")
    QUESTION_IMAGE = {}
    if os.path.exists(question_image_path):
        with open(question_image_path, "r") as f:
            QUESTION_IMAGE = json.load(f)
    else:
        for k, v in QUESTION_META_DATA.items():
            QUESTION_IMAGE[k] = re.findall(r"question_\d+-image_\d+", v["content"])
        write_json(QUESTION_IMAGE, question_image_path)

    analysis_image_path = os.path.join(OUTPUT_DIR, "analysis_image.json")
    ANALYSIS_IMAGE = {}
    if os.path.exists(analysis_image_path):
        with open(analysis_image_path, "r") as f:
            ANALYSIS_IMAGE = json.load(f)
    else:
        for k, v in QUESTION_META_DATA.items():
            ANALYSIS_IMAGE[k] = re.findall(r"analysis_\d+-image_\d+", v["analysis"])
        write_json(ANALYSIS_IMAGE, analysis_image_path)

    question_content_path = os.path.join(OUTPUT_DIR, "question_content.json")
    QUESTION_CONTENT = {}
    if os.path.exists(question_content_path):
        with open(question_content_path, "r") as f:
            QUESTION_CONTENT = json.load(f)
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

    # 翻译为英文
    question_translation = {}
    question_translation_path = os.path.join(OUTPUT_DIR, "question_translation.json")
    if os.path.exists(question_translation_path):
        with open(question_translation_path, "r") as f:
            question_translation_ = json.load(f)
            # 排一下序
            q_ids = sorted(list(map(int, question_translation_.keys())))
            for q_id in q_ids:
                question_translation[str(q_id)] = question_translation_[str(q_id)]
    num_translated = 0
    for q_id, q_content in QUESTION_CONTENT.items():
        if num_translated >= NUM_CALL_API:
            break

        if q_id in question_translation:
            continue

        q_text = q_content["content_with_image_name"]
        question_translation[q_id] = {
            "chinese": q_text
        }
        q_type = "fill_up" if q_content["type"] == "填空" else "multi_choice"
        translated = translate_question2english(MODEL_NAME, q_type, q_text)
        question_translation[q_id]["english"] = translated.content
        num_translated += 1
    write_json(question_translation, question_translation_path)
