from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import re

from lib.util.data import load_json

DATA_DIR = r"/Users/dream/Desktop/code/projects/dlkt/lab/dataset_preprocessed/xes3g5m"
QUESTION_META = load_json(f"{DATA_DIR}/question_meta.json")
CONCEPT_META = load_json(f"{DATA_DIR}/concept_meta.json")

QUESTION_IMAGE = {}
for k, v in QUESTION_META.items():
    QUESTION_IMAGE[k] = re.findall(r"question_\d+-image_\d+", v["content"])

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
QUESTION_CONTENT = {}
for k, v in QUESTION_META.items():
    QUESTION_CONTENT[k] = {
        "content": v["content"],
        "image": QUESTION_IMAGE[k],
        "type": v["type"]
    }
    for image_str in QUESTION_IMAGE[k]:
        QUESTION_CONTENT[k]["content"] = QUESTION_CONTENT[k]["content"].replace(image_str, "")
    QUESTION_CONTENT[k]["content"] = QUESTION_CONTENT[k]["content"].strip()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt(chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
    # print("chatgpt message",chatgpt_messages)
    url = "https://www.jiujiuai.life/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer sk-XHx1xQu7iM5kmgYVFd08Ce8f3e5a4dEa8c2c00Bf62225244"
        "Authorization": "sk-XHx1xQu7iM5kmgYVFd08Ce8f3e5a4dEa8c2c00Bf62225244"
    }

    data = {
        "model": model,
        "messages": chatgpt_messages,
        "max_tokens": max_tokens
    }
    response = requests.post(url, json=data, headers=headers)
    response_data = response.json()
    reply = response_data['choices'][0]['message']['content']
    total_tokens = response_data['usage']['total_tokens']

    return reply, total_tokens


def ask_llm_kt(question_seq, correct_seq):
    question_seq = list(map(str, question_seq))
    question_content_seq = list(map(lambda x: QUESTION_META[x]["content"], question_seq))
    concept_content_seq = list(map(
        lambda x: "; ".join(QUESTION_META[x]["kc_routes"]),
        question_seq
    ))
    correct_seq = list(map(lambda x: "做对" if x else "做错", correct_seq))

    example_seq_len = 2
    example_prompt1 = f"[`{', '.join([f'({question_content_seq[i]}`, `{concept_content_seq[i]}`, `{correct_seq[i]}`)' for i in range(example_seq_len)])}]"
    example_prompt1 += f", Please determine whether Student A can do this exercise correctly: {question_content_seq[example_seq_len + 1]}\n"

    seq_len = 3
    qa_prompt = f"[`{', '.join([f'({question_content_seq[i]}`, `{concept_content_seq[i]}`, `{correct_seq[i]}`)' for i in range(seq_len)])}]"
    qa_prompt += f", Please determine whether Student A can do this exercise correctly: {question_content_seq[seq_len + 1]}\n"

    prompt = "Now you are an agent of an intelligent tutoring system.\n" \
             "You need to complete a task called knowledge tracking, " \
             "which is to predict whether the student can answer a specified question correctly " \
             "based on the student's historical exercise answer record, " \
             "and provide a basis for judgment.\n" \
             "Now tell you the format of students’ historical practice records: " \
             "[(`Exercise 1 text`, `Knowledge concepts examined in Exercise 1`, `Exercise results`), " \
             "(`Exercise 2 text`, `Knowledge concepts examined in Exercise 2`, `Exercise results`), ...]\n" \
             "example:\n" \
             f"input: {example_prompt1}\n" \
             f"output: I think Student A can do this exercise correctly. The reasons are as follows,\n" \
             f"Student A got the first exercise and the second exercise right one after another. " \
             f"The knowledge concepts examined in these two exercises are all one-stroke judgments in " \
             f"graph theory problems. This shows that the student has a good grasp of this knowledge point. " \
             f"The knowledge concept corresponding to the exercise you asked is the multiple strokes in the graph " \
             f"theory problem. Multiple strokes and one-stroke exercises are highly correlated, that is, " \
             f"a multi-stroke exercise can be converted into a one-stroke exercise. " \
             f"Therefore, I judged that Student A could do this exercise correctly.\n" \
             f"input: {qa_prompt}\n" \
             f"output: "

    print(prompt)

    # message = [{"role": "user", "content": prompt}]
    # # 注意max_tokens不能太大
    # question, n_tokens = call_chatgpt(message, model="gpt-3.5-turbo", max_tokens=10000)
    # print(question)


def translate_question(question_id):
    question_content = QUESTION_CONTENT[str(question_id)]

    if question_content["type"] == "填空":
        prompt = \
            "Please translate this exercise into English.\n" \
            "Please note: (1) Do not translate the `$` symbol; (2) This question is a fill-in-the-blank " \
            "question, but the part that needs to be filled in is not blank. Please use 3 underlines (i.e. ___) to " \
            "mark the places where the respondent needs to fill in the blank.\n" \
            "input: 学校有舞蹈，唱歌、围棋、绘画四种兴趣班． 小宇、小明、小丽三个小朋友准备报名，每人只能报一个班而且各不相同，一共有种不同的报名方法．\n" \
            "output: The school has four interest classes: dance, singing, go, and painting. " \
            "Three children, Xiaoyu, Xiaoming and Xiaoli, are ready to register. " \
            "Each of them can only register for one class and each class is different. " \
            "There are ___ different registration methods.\n" \
            "input: 书架上有 $$2$$ 本不同的英语书，$$4$$ 本不同的语文书，$$3$$ 本不同的数学书． 现在要从中取出$$2$$ 本，而且不能是同一科的，一共有种不同取法．\n" \
            "output: There are $$2$$ different English books, $$4$$ different Chinese books, and $$3$$ different " \
            "math books on the bookshelf. Now we need to take out $$2$$ books from it, and they cannot be from the " \
            "same subject. There are ___ different ways to take them.\n" \
            f"input: {question_content['content']}\n" \
            "output: "
    else:
        prompt = \
            "Please translate this exercise into English.\n" \
            "Please note that do not translate the `$` symbol.\n" \
            "input: 有一个边长为$$5$$米的正方形花坛，在外围四周铺$$1$$米宽的小路，小路的面积是（ ）平方米．\n" \
            "output: There is a square flower bed with a side length of $$5$$ meters. " \
            "A $$1$$ meter wide path is paved around the outside. The area of the path is ( ) square meters.\n" \
            "input: '兔子的胡萝卜丢了，去找黑猫警长报案，黑猫警长最终锁定了四只动物，分别是狐狸、老虎、狮子、狼，罪犯就在它们之中．\n" \
            "狐狸说：胡萝卜不是我偷的；\n老虎说：胡萝卜是狼偷的；\n狮子说：胡萝卜是老虎偷的；\n狼说：我没有偷过胡萝卜．\n" \
            "后经了解，四只动物中只有一只动物说的是真话．请问：胡萝卜是谁偷的？（ ）'" \
            "output: The rabbit lost his carrot and went to the Black Cat Sheriff to report the crime. " \
            "The Black Cat Sheriff finally targeted four animals, namely the fox, tiger, lion, and wolf, " \
            "and the criminal was among them.\n" \
            "The fox said: I didn’t steal the carrot;\n" \
            "The tiger said: The carrot was stolen by the wolf;\n" \
            "The lion said: The tiger stole the carrot;\n" \
            "The wolf said: I have never stolen a carrot.\n" \
            "It was later learned that only one of the four animals was telling the truth. " \
            "Please ask: Who stole the carrots? ( )\n" \
            f"input: {question_content['content']}\n" \
            "output: "

    message = [{"role": "user", "content": prompt}]
    question, n_tokens = call_chatgpt(message, model="gpt-3.5-turbo", max_tokens=3000)
    print(prompt)
    print(question)


if __name__ == "__main__":
    # ask_llm_kt([3751, 3752, 3753, 3754, 1990, 3739, 3740, 3742, 3756, 3757], [1, 1, 1, 0, 1, 1, 1, 1, 0, 0])
    # 单选题：20,30,32,34,48,49
    # translate_question(3)
    # translate_question(4)
    # translate_question(5)

    # 结果如下
    # Please translate this exercise into English.
    # Please note: (1) Do not translate the `$` symbol; (2) This question is a fill-in-the-blank question, but the part that needs to be filled in is not blank. Please use 3 underlines (i.e. ___) to mark the places where the respondent needs to fill in the blank.
    # input: 学校有舞蹈，唱歌、围棋、绘画四种兴趣班． 小宇、小明、小丽三个小朋友准备报名，每人只能报一个班而且各不相同，一共有种不同的报名方法．
    # output: The school has four interest classes: dance, singing, go, and painting. Three children, Xiaoyu, Xiaoming and Xiaoli, are ready to register. Each of them can only register for one class and each class is different. There are ___ different registration methods.
    # input: 书架上有 $$2$$ 本不同的英语书，$$4$$ 本不同的语文书，$$3$$ 本不同的数学书． 现在要从中取出$$2$$ 本，而且不能是同一科的，一共有种不同取法．
    # output: There are $$2$$ different English books, $$4$$ different Chinese books, and $$3$$ different math books on the bookshelf. Now we need to take out $$2$$ books from it, and they cannot be from the same subject. There are ___ different ways to take them.
    # input: 用四种颜色去涂如图所示的三块区域，要求相邻的区域涂不同的颜色，那么共有种不同的涂法.
    # output:
    # Use four colors to paint the three regions shown in the figure. Adjacent regions should be painted with different colors. There are ___ different ways to paint them.

    # Please translate this exercise into English.
    # Please note: (1) Do not translate the `$` symbol; (2) This question is a fill-in-the-blank question, but the part that needs to be filled in is not blank. Please use 3 underlines (i.e. ___) to mark the places where the respondent needs to fill in the blank.
    # input: 学校有舞蹈，唱歌、围棋、绘画四种兴趣班． 小宇、小明、小丽三个小朋友准备报名，每人只能报一个班而且各不相同，一共有种不同的报名方法．
    # output: The school has four interest classes: dance, singing, go, and painting. Three children, Xiaoyu, Xiaoming and Xiaoli, are ready to register. Each of them can only register for one class and each class is different. There are ___ different registration methods.
    # input: 书架上有 $$2$$ 本不同的英语书，$$4$$ 本不同的语文书，$$3$$ 本不同的数学书． 现在要从中取出$$2$$ 本，而且不能是同一科的，一共有种不同取法．
    # output: There are $$2$$ different English books, $$4$$ different Chinese books, and $$3$$ different math books on the bookshelf. Now we need to take out $$2$$ books from it, and they cannot be from the same subject. There are ___ different ways to take them.
    # input: 按下表给出的词造句，每句必须包括一个人、一个交通工具，以及一个目的地，请问可以造出个不同的句子．爸爸乘飞机去北京妈妈火车拉萨我汽车台北
    # output:
    # Using the words given in the table, make a sentence that must include a person, a mode of transportation, and a destination. Please tell us how many different sentences can be made. "Dad takes a plane to Beijing. Mom takes a train to Lhasa. I take a car to Taipei." There are ___ different sentences that can be made.

    # Please translate this exercise into English.
    # Please note: (1) Do not translate the `$` symbol; (2) This question is a fill-in-the-blank question, but the part that needs to be filled in is not blank. Please use 3 underlines (i.e. ___) to mark the places where the respondent needs to fill in the blank.
    # input: 学校有舞蹈，唱歌、围棋、绘画四种兴趣班． 小宇、小明、小丽三个小朋友准备报名，每人只能报一个班而且各不相同，一共有种不同的报名方法．
    # output: The school has four interest classes: dance, singing, go, and painting. Three children, Xiaoyu, Xiaoming and Xiaoli, are ready to register. Each of them can only register for one class and each class is different. There are ___ different registration methods.
    # input: 书架上有 $$2$$ 本不同的英语书，$$4$$ 本不同的语文书，$$3$$ 本不同的数学书． 现在要从中取出$$2$$ 本，而且不能是同一科的，一共有种不同取法．
    # output: There are $$2$$ different English books, $$4$$ different Chinese books, and $$3$$ different math books on the bookshelf. Now we need to take out $$2$$ books from it, and they cannot be from the same subject. There are ___ different ways to take them.
    # input: 下图是一个变形的蘑菇，一共分为六块区域．现在要用四种颜色对其染色，要求相邻的两块区域（有公共边的两块区域称为相邻）染成不同的颜色．如果颜色能反复使用，那么一共有种不同的染色方法．
    # output:
    # The picture below is a deformed mushroom, divided into six regions. Now we want to color it with four different colors, and the adjacent regions (two regions with a common edge are called adjacent) should be colored in different colors. If colors can be used repeatedly, there are ___ different coloring methods in total.

    print("")

