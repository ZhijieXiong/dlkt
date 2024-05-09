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

    # 保存输出的地址（定了以后不要变更，为了稳定以及节约成本，该脚本会运行多次，如每次翻译n道题，所以需要读取之前已做过的操作）
    OUTPUT_DIR = r"F:\code\myProjects\dlkt\lab\math_dataset\xes3g5m"
    QUESTION_CONTENT_PATH = r"F:\code\myProjects\dlkt\lab\math_dataset\xes3g5m\question_content.json"
    QUESTION_CONTENT = {}
    if os.path.exists(QUESTION_CONTENT_PATH):
        QUESTION_CONTENT = load_json(QUESTION_CONTENT_PATH)

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
