import re
import base64
import os

from lib.util.data import load_json


# 多张图片习题 [('522', 2), ('567', 2), ('807', 2), ('810', 3), ('914', 2), ('974', 4), ('983', 5), ('1126', 2), ('1328',
# 3), ('1370', 4), ('1792', 2), ('2056', 2), ('2151', 3), ('2275', 3), ('2316', 4), ('2318', 2), ('2359', 2),
# ('2368', 2), ('2656', 3), ('3116', 5), ('3151', 3), ('3252', 3), ('3264', 2), ('3355', 4), ('3378', 2), ('3457',
# 3), ('3472', 2), ('3585', 2), ('3726', 2), ('3808', 2), ('3810', 3), ('3833', 4), ('3889', 8), ('4247', 2),
# ('4354', 4), ('4610', 2), ('4611', 4), ('4752', 2), ('5035', 2), ('5070', 2), ('5117', 4), ('5192', 2), ('5193',
# 2), ('5387', 2), ('5794', 6), ('5941', 2), ('5984', 2), ('6052', 4), ('6149', 2), ('6176', 3), ('6253', 2),
# ('6383', 2), ('6406', 2), ('6426', 2), ('6515', 2), ('6609', 3), ('6682', 2), ('6695', 5), ('6700', 2), ('6893',
# 2), ('7154', 2), ('7245', 2), ('7263', 2), ('7264', 2), ('7286', 4), ('7455', 4), ('7467', 3), ('7516', 2),
# ('7584', 2), ('7621', 4), ('7632', 2)]
# 如习题5794： '艾迪想买一份主食，再搭配一种饮品，一共有种不同的买法．
#
#
#
# 主食
# 饮品
#
#
#  question_5794-image_0 question_5794-image_1
#  question_5794-image_2 question_5794-image_3 question_5794-image_4 question_5794-image_5'


DATA_DIR = r"/Users/dream/Desktop/code/projects/dlkt/lab/dataset_preprocessed/xes3g5m"
IMAGE_DIR = r"/Users/dream/Desktop/code/projects/dlkt/lab/dataset_raw/xes3g5m/metadata/images"
QUESTION_META = load_json(f"{DATA_DIR}/question_meta.json")

QUESTION_IMAGE = {}
for k, v in QUESTION_META.items():
    QUESTION_IMAGE[k] = re.findall(r"question_\d+-image_\d+", v["content"])

ANALYSIS_IMAGE = {}
for k, v in QUESTION_META.items():
    ANALYSIS_IMAGE[k] = re.findall(r"analysis_\d+-image_\d+", v["analysis"])

QUESTION_CONTENT = {}
for k, v in QUESTION_META.items():
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
        QUESTION_CONTENT[k]["content_with_image_name"] = QUESTION_CONTENT[k]["content_with_image_name"].replace(image_str, f"[%{image_str}%]")
    QUESTION_CONTENT[k]["content_only_text"] = QUESTION_CONTENT[k]["content_only_text"].strip()
    QUESTION_CONTENT[k]["content_with_image_name"] = QUESTION_CONTENT[k]["content_with_image_name"].strip()

    for image_str in ANALYSIS_IMAGE[k]:
        QUESTION_CONTENT[k]["analysis_only_text"] = QUESTION_CONTENT[k]["analysis_only_text"].replace(image_str, "")
        QUESTION_CONTENT[k]["analysis_with_image_name"] = QUESTION_CONTENT[k]["analysis_with_image_name"].replace(image_str, f"[%{image_str}%]")

    QUESTION_CONTENT[k]["analysis_only_text"] = QUESTION_CONTENT[k]["analysis_only_text"].strip()
    QUESTION_CONTENT[k]["analysis_with_image_name"] = QUESTION_CONTENT[k]["analysis_with_image_name"].strip()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def construct_vision_messages(text, image_dir):
    """
    构造prompt，用于多模态模型
    :param text: 用[%image_path%]表示图片对应的位置
    :param image_dir:
    :return: api所需要的格式，即message中的content列表，格式为[{"type": "text", "text": "what is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}}, ]
    """
    if "question" in text and "image" in text:
        # 用正则表达式划分
        # re.split(r'\[%(question_\d+-image_\d+)%\]',
        #          "如图[%question_5-image_1%]所示，请问里面有几个人；如果是[%question_5-image_2%]呢？")
        split_result = re.split(r'\[%(question_\d+-image_\d+)%\]', text)
    else:
        split_result = [text]
    messages = []
    for item in split_result:
        if item == "":
            continue
        if "question" in item and "image" in item:
            messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(os.path.join(image_dir, item + '.png'))}"
                }
            })
            continue
        messages.append({
            "type": "text",
            "text": item
        })
    return messages


for k, v in QUESTION_CONTENT.items():
    QUESTION_CONTENT[k]["messages4api_chinese"] = construct_vision_messages(v["content_with_image_name"], IMAGE_DIR)



