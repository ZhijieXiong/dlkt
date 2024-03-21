from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import tiktoken

from xes3g5m_process import *
from keys import jiujiuai_api_key

# cl100k_base: gpt-4, gpt-3.5-turbo, text-embedding-ada-002
# p50k_base: Codex models, text-davinci-002, text-davinci-003
# r50k_base (or gpt2): GPT-3 models like davinci
ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt(chatgpt_messages, max_tokens=4000, model="gpt-3.5-turbo"):
    url = "https://www.jiujiuai.life/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer sk-XHx1xQu7iM5kmgYVFd08Ce8f3e5a4dEa8c2c00Bf62225244"
        "Authorization": jiujiuai_api_key
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


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt_vision(chatgpt_messages, max_tokens=4000):
    url = "https://www.jiujiuai.life/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer sk-XHx1xQu7iM5kmgYVFd08Ce8f3e5a4dEa8c2c00Bf62225244"
        "Authorization": jiujiuai_api_key
    }

    data = {
        "model": "gpt-4-vision-preview",
        "messages": chatgpt_messages,
        "max_tokens": max_tokens
    }
    response = requests.post(url, json=data, headers=headers)
    response_data = response.json()
    reply = response_data['choices'][0]['message']['content']
    total_tokens = response_data['usage']['total_tokens']

    return reply, total_tokens


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt_embedding(text, max_tokens=8000):
    url = "https://www.jiujiuai.life/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer sk-XHx1xQu7iM5kmgYVFd08Ce8f3e5a4dEa8c2c00Bf62225244"
        "Authorization": jiujiuai_api_key
    }

    data = {
        "model": "text-embedding-ada-002",
        "input": text,
        "encoding_format": "float",
        "max_tokens": max_tokens,
    }
    response = requests.post(url, json=data, headers=headers)
    response_data = response.json()
    if type(text) is list:
        return response_data['data']
    else:
        return response_data['data'][0]


def ask_llm_kt(question_seq, correct_seq):
    question_seq = list(map(str, question_seq))
    question_content_seq = list(map(lambda x: QUESTION_CONTENT[x]["content_only_text"], question_seq))
    concept_content_seq = list(map(
        lambda x: "; ".join(QUESTION_CONTENT[x]["kc_routes"]),
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
             f"graph theory problems. This shows that the student has a good grasp of this knowledge concept. " \
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
            "Please note: (1) Please be careful not to translate the `$` symbol, `[%` and `%]` symbols. In addition, " \
            "please do not translate the content in `$$` and the content in `[%%]`; (2) This question is a fill-in-the-blank " \
            "question, but the part that needs to be filled in is not blank. Please use 3 underlines (i.e. ___) to " \
            "mark the places where the respondent needs to fill in the blank.\n" \
            "Please be careful to avoid grammatical errors in the translated content.\n" \
            "Here are some examples:\n" \
            "input: 学校有舞蹈，唱歌、围棋、绘画四种兴趣班． 小宇、小明、小丽三个小朋友准备报名，每人只能报一个班而且各不相同，一共有种不同的报名方法．\n" \
            "output: This is a fill-in-the-blank question. The school has four interest classes: dance, singing, go, and painting. " \
            "Three children, Xiaoyu, Xiaoming and Xiaoli, are ready to register. " \
            "Each of them can only register for one class and each class is different. " \
            "There are ___ different registration methods.\n" \
            "input: 书架上有 $$2$$ 本不同的英语书，$$4$$ 本不同的语文书，$$3$$ 本不同的数学书． 现在要从中取出$$2$$ 本，而且不能是同一科的，一共有种不同取法．\n" \
            "output: This is a fill-in-the-blank question. There are $$2$$ different English books, $$4$$ different Chinese books, and $$3$$ different " \
            "math books on the bookshelf. Now we need to take out $$2$$ books from it, and they cannot be from the " \
            "same subject. There are ___ different ways to take them.\n" \
            f"input: {question_content['content_with_image_name']}\n" \
            "output: "
    else:
        prompt = \
            "Please translate this exercise into English.\n" \
            "Please note: (1) Please be careful not to translate the `$` symbol, `[%` and `%]` symbols. In addition, " \
            "please do not translate the content in `$$` and the content in `[%%]`; (2) This question is a multiple choice " \
            "question, please keep the brackets used to fill in the options.\n" \
            "Please be careful to avoid grammatical errors in the translated content.\n" \
            "Here are some examples:\n" \
            "input: 有一个边长为$$5$$米的正方形花坛，在外围四周铺$$1$$米宽的小路，小路的面积是（ ）平方米．\n" \
            "output: This is a multiple choice question. There is a square flower bed with a side length of $$5$$ meters. " \
            "A $$1$$ meter wide path is paved around the outside. The area of the path is ( ) square meters.\n" \
            "input: 兔子的胡萝卜丢了，去找黑猫警长报案，黑猫警长最终锁定了四只动物，分别是狐狸、老虎、狮子、狼，罪犯就在它们之中．\n" \
            "狐狸说：胡萝卜不是我偷的；\n老虎说：胡萝卜是狼偷的；\n狮子说：胡萝卜是老虎偷的；\n狼说：我没有偷过胡萝卜．\n" \
            "后经了解，四只动物中只有一只动物说的是真话．请问：胡萝卜是谁偷的？（ ）" \
            "output: This is a multiple choice question. The rabbit lost his carrot and went to the Black Cat Sheriff to report the crime. " \
            "The Black Cat Sheriff finally targeted four animals, namely the fox, tiger, lion, and wolf, " \
            "and the criminal was among them.\n" \
            "The fox said: I didn’t steal the carrot;\n" \
            "The tiger said: The carrot was stolen by the wolf;\n" \
            "The lion said: The tiger stole the carrot;\n" \
            "The wolf said: I have never stolen a carrot.\n" \
            "It was later learned that only one of the four animals was telling the truth. " \
            "Please ask: Who stole the carrots? ( )\n" \
            f"input: {question_content['content_with_image_name']}\n" \
            "output: "

    message = [{"role": "user", "content": prompt}]
    question, n_tokens = call_chatgpt(message, model="gpt-3.5-turbo", max_tokens=3000)
    print(prompt)
    print(question)


def ask_llm_question_vision():
    # 以第5题为例
    prompt = \
        "What is the answer to this math question?\n" \
        ""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is a fill-in-the-blank question. The figure below is a deformed mushroom divided into six areas. Now we need to color it with four colors, and the adjacent two areas (the two areas with a common edge are called adjacent) must be colored with different colors. If the colors can be used repeatedly, there are ___ different coloring methods."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image('/Users/dream/Desktop/code/projects/dlkt/lab/dataset_raw/xes3g5m/metadata/images/question_5-image_0.png')}"
                    }
                }
            ]
        }
    ]
    question, n_tokens = call_chatgpt_vision(messages, max_tokens=3000)
    print(prompt)
    print(question)


if __name__ == "__main__":
    pass
    # 知识追踪，但是token数太多
    # ask_llm_kt([3751, 3752, 3753, 3754, 1990, 3739, 3740, 3742, 3756, 3757], [1, 1, 1, 0, 1, 1, 1, 1, 0, 0])

    # 填空题
    # translate_question(5)
    # 翻译结果
    # Please translate this exercise into English.
    # Please note: (1) Please be careful not to translate the `$` symbol, `[%` and `%]` symbols. In addition, please do not translate the content in `$$` and the content in `[%%]`; (2) This question is a fill-in-the-blank question, but the part that needs to be filled in is not blank. Please use 3 underlines (i.e. ___) to mark the places where the respondent needs to fill in the blank.
    # Please be careful to avoid grammatical errors in the translated content.
    # Here are some examples:
    # input: 学校有舞蹈，唱歌、围棋、绘画四种兴趣班． 小宇、小明、小丽三个小朋友准备报名，每人只能报一个班而且各不相同，一共有种不同的报名方法．
    # output: This is a fill-in-the-blank question. The school has four interest classes: dance, singing, go, and painting. Three children, Xiaoyu, Xiaoming and Xiaoli, are ready to register. Each of them can only register for one class and each class is different. There are ___ different registration methods.
    # input: 书架上有 $$2$$ 本不同的英语书，$$4$$ 本不同的语文书，$$3$$ 本不同的数学书． 现在要从中取出$$2$$ 本，而且不能是同一科的，一共有种不同取法．
    # output: This is a fill-in-the-blank question. There are $$2$$ different English books, $$4$$ different Chinese books, and $$3$$ different math books on the bookshelf. Now we need to take out $$2$$ books from it, and they cannot be from the same subject. There are ___ different ways to take them.
    # input: 下图是一个变形的蘑菇，一共分为六块区域．现在要用四种颜色对其染色，要求相邻的两块区域（有公共边的两块区域称为相邻）染成不同的颜色．如果颜色能反复使用，那么一共有种不同的染色方法． [%question_5-image_0%]
    # output:
    # This is a fill-in-the-blank question. The figure below is a deformed mushroom divided into six areas. Now we need to color it with four colors, and the adjacent two areas (the two areas with a common edge are called adjacent) must be colored with different colors. If the colors can be used repeatedly, there are ___ different coloring methods. [%question_5-image_0%]

    # 单选题：20,30,32,34,48,49
    # translate_question(20)
    # 翻译结果
    # Please translate this exercise into English.
    # Please note: (1) Please be careful not to translate the `$` symbol, `[%` and `%]` symbols. In addition, please do not translate the content in `$$` and the content in `[%%]`; (2) This question is a multiple choice question, please keep the brackets used to fill in the options.
    # Please be careful to avoid grammatical errors in the translated content.
    # Here are some examples:
    # input: 有一个边长为$$5$$米的正方形花坛，在外围四周铺$$1$$米宽的小路，小路的面积是（ ）平方米．
    # output: This is a multiple choice question. There is a square flower bed with a side length of $$5$$ meters. A $$1$$ meter wide path is paved around the outside. The area of the path is ( ) square meters.
    # input: 兔子的胡萝卜丢了，去找黑猫警长报案，黑猫警长最终锁定了四只动物，分别是狐狸、老虎、狮子、狼，罪犯就在它们之中．
    # 狐狸说：胡萝卜不是我偷的；
    # 老虎说：胡萝卜是狼偷的；
    # 狮子说：胡萝卜是老虎偷的；
    # 狼说：我没有偷过胡萝卜．
    # 后经了解，四只动物中只有一只动物说的是真话．请问：胡萝卜是谁偷的？（ ）output: This is a multiple choice question. The rabbit lost his carrot and went to the Black Cat Sheriff to report the crime. The Black Cat Sheriff finally targeted four animals, namely the fox, tiger, lion, and wolf, and the criminal was among them.
    # The fox said: I didn’t steal the carrot;
    # The tiger said: The carrot was stolen by the wolf;
    # The lion said: The tiger stole the carrot;
    # The wolf said: I have never stolen a carrot.
    # It was later learned that only one of the four animals was telling the truth. Please ask: Who stole the carrots? ( )
    # input: 有一个边长为$$5$$米的正方形花坛，在外围四周铺$$1$$米宽的小路，小路的面积是（ ）平方米．
    # [%question_20-image_0%]
    # output:
    # This is a multiple choice question. There is a square flower bed with a side length of $$5$$ meters. A $$1$$ meter wide path is paved around the outside. The area of the path is ( ) square meters.
    # [%question_20-image_0%]

    # ask_llm_question_vision()

    emb = call_chatgpt_embedding("this is a math exercise.")
    # 返回格式: {'object': 'embedding', 'index': 0, 'embedding': [0.012580604, ...]}
    embs = call_chatgpt_embedding(["this is a math exercise.", "你好，世界。"])
    # 返回格式：[{'object': 'embedding', 'index': 0, 'embedding': [0.012580604, ...]}, {'object': 'embedding', 'index': 0, 'embedding': [0.012580604, ...]}, ...]
    print(emb)
    print(embs)
