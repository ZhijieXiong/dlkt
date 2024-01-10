from keys import openai_key as OPENAI_KEY

from xes3g5m_process import *

from openai import OpenAI


def demo():
    client = OpenAI(api_key=OPENAI_KEY)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
            },
            {
                "role": "user",
                "content": "Compose a poem that explains the concept of recursion in programming."
            }
        ]
    )

    print(completion.choices[0].message)


def call_chatgpt(chatgpt_messages):
    client = OpenAI(api_key=OPENAI_KEY)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chatgpt_messages
    )

    for i in range(len(chatgpt_messages)):
        print(f"prompt {i}:")
        print(chatgpt_messages[i]["content"])
        print(f"\noutput {i}:")
        print(completion.choices[i].message.content)


def demo_kt(ask_question_seq, ask_correct_seq, use_question_content=False):
    example_question_seq = [3751, 3752, 3753, 3754, 1990, 3739, 3740, 3742, 3756, 3757]
    example_correct_seq = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
    example_question_seq = list(map(str, example_question_seq))
    example_question_content_seq = list(map(lambda x: QUESTION_CONTENT[x]["content_only_text"], example_question_seq))
    example_concept_content_seq = list(map(
        lambda x: "; ".join(QUESTION_CONTENT[x]["kc_routes"]),
        example_question_seq
    ))
    example_correct_seq = list(map(lambda x: "做对" if x else "做错", example_correct_seq))

    example_seq_lens = [2, 8]
    example_prompts = []
    for example_seq_len in example_seq_lens:
        if use_question_content:
            example_prompt = f"[`{', '.join([f'(`{example_question_content_seq[i]}`, `{example_concept_content_seq[i]}`, `{example_correct_seq[i]}`)' for i in range(example_seq_len)])}]"
        else:
            example_prompt = f"[`{', '.join([f'(`{example_concept_content_seq[i]}`, `{example_correct_seq[i]}`)' for i in range(example_seq_len)])}]"
        example_prompt += f", Please determine whether Student A can do this exercise correctly: {example_question_content_seq[example_seq_len]}\n"
        example_prompts.append(example_prompt)
    example_outputs = [
        "output: I think Student A can do this exercise correctly. The reasons are as follows,\n"
        "Student A got the first exercise and the second exercise right one after another. "
        "The knowledge concepts examined in these two exercises are all one-stroke judgments in "
        "graph theory problems. This shows that the student has a good grasp of this knowledge concept. "
        "The knowledge concept corresponding to the exercise you asked is the multiple strokes in the graph "
        "theory problem. Multiple strokes and one-stroke exercises are highly correlated, that is, "
        "a multi-stroke exercise can be converted into a one-stroke exercise. "
        "Therefore, I judged that Student A could do this exercise correctly.\n\n",

        "output: I think Student A can not do this exercise correctly. The reasons are as follows,\n"
        "Based on the student's historical exercise answer record, "
        "we can observe that the student has a good grasp of the knowledge concepts "
        "examined in the exercises related to graph theory problems, "
        "such as one-stroke judgments and multiple strokes converting into one-stroke exercises. "
        "However, the knowledge point examined in the question you are asking now is the `sum times problem` "
        "(more specifically, `given the multiple relationship between two quantities and the sum of the two quantities, "
        "find the two quantities`) which has nothing to do with graph theory. "
        "Therefore, I guessed that it was the first time for Student A to do this type of question. "
        "So I judged that Student A would do this question wrong.\n\n"

    ]
    example_prompt_all = ""
    for example_prompt, example_output in zip(example_prompts, example_outputs):
        example_prompt_all += f"input: {example_prompt}"
        example_prompt_all += example_output

    ask_question_seq = list(map(str, ask_question_seq))
    ask_question_content_seq = list(map(lambda x: QUESTION_CONTENT[x]["content_only_text"], ask_question_seq))
    ask_concept_content_seq = list(map(
        lambda x: "; ".join(QUESTION_CONTENT[x]["kc_routes"]),
        ask_question_seq
    ))
    ask_correct_seq = list(map(lambda x: "做对" if x else "做错", ask_correct_seq))
    ask_seq_len = 9
    if use_question_content:
        qa_prompt = f"[`{', '.join([f'(`{ask_question_content_seq[i]}`, `{ask_concept_content_seq[i]}`, `{ask_correct_seq[i]}`)' for i in range(ask_seq_len)])}]"
    else:
        qa_prompt = f"[`{', '.join([f'(`{ask_concept_content_seq[i]}`, `{ask_correct_seq[i]}`)' for i in range(ask_seq_len)])}]"
    qa_prompt += f", Please determine whether Student A can do this exercise correctly: {ask_question_content_seq[ask_seq_len]}\n"

    prompt = "Now you are an agent of an intelligent tutoring system.\n" \
             "You need to complete a task called knowledge tracking, " \
             "which is to predict whether the student can answer a specified question correctly " \
             "based on the student's historical exercise answer record, " \
             "and provide a basis for judgment.\n" \
             "Now tell you the format of students’ historical practice records: " \
             f"[({'`Exercise 1 text`, ' if use_question_content else ''}`Knowledge concepts examined in Exercise 1`, `Exercise results`), " \
             f"({'`Exercise 2 text`, ' if use_question_content else ''}`Knowledge concepts examined in Exercise 2`, `Exercise results`), ...]\n" \
             "examples:\n" \
             f"{example_prompt_all}\n" \
             f"input: {qa_prompt}\n" \
             f"output: "

    message = [{"role": "user", "content": prompt}]
    call_chatgpt(message)


def demo_question_rec(question_seq, correct_seq, use_question_content=False):
    example_question_seq = [3751, 3752, 3753, 3754, 1990, 3739, 3740, 3742, 3756, 3757]
    example_correct_seq = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
    example_question_seq = list(map(str, example_question_seq))
    example_question_content_seq = list(map(lambda x: QUESTION_CONTENT[x]["content_only_text"], example_question_seq))
    example_concept_content_seq = list(map(
        lambda x: "; ".join(QUESTION_CONTENT[x]["kc_routes"]),
        example_question_seq
    ))
    example_correct_seq = list(map(lambda x: "做对" if x else "做错", example_correct_seq))

    question_seq = list(map(str, question_seq))
    question_content_seq = list(map(lambda x: QUESTION_CONTENT[x]["content_only_text"], question_seq))
    concept_content_seq = list(map(
        lambda x: "; ".join(QUESTION_CONTENT[x]["kc_routes"]),
        question_seq
    ))
    correct_seq = list(map(lambda x: "做对" if x else "做错", correct_seq))

    example_seq_len = 2
    if use_question_content:
        example_prompt1 = f"[`{', '.join([f'(`{question_content_seq[i]}`, `{concept_content_seq[i]}`, `{correct_seq[i]}`)' for i in range(example_seq_len)])}]"
    else:
        example_prompt1 = f"[`{', '.join([f'(`{concept_content_seq[i]}`, `{correct_seq[i]}`)' for i in range(example_seq_len)])}]"
    example_prompt1 += f", Please determine whether Student A can do this exercise correctly: {question_content_seq[example_seq_len]}\n"

    seq_len = 9
    if use_question_content:
        qa_prompt = f"[`{', '.join([f'(`{question_content_seq[i]}`, `{concept_content_seq[i]}`, `{correct_seq[i]}`)' for i in range(seq_len)])}]"
    else:
        qa_prompt = f"[`{', '.join([f'(`{concept_content_seq[i]}`, `{correct_seq[i]}`)' for i in range(seq_len)])}]"
    qa_prompt += f", Please determine whether Student A can do this exercise correctly: {question_content_seq[seq_len]}\n"

    prompt = "Now you are an agent of an intelligent question recommendation system.\n" \
             "You need to complete a task called knowledge tracking, " \
             "which is to predict whether the student can answer a specified question correctly " \
             "based on the student's historical exercise answer record, " \
             "and provide a basis for judgment.\n" \
             "Now tell you the format of students’ historical practice records: " \
             f"[({'`Exercise 1 text`, ' if use_question_content else ''}`Knowledge concepts examined in Exercise 1`, `Exercise results`), " \
             f"({'`Exercise 2 text`, ' if use_question_content else ''}`Knowledge concepts examined in Exercise 2`, `Exercise results`), ...]\n" \
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

    message = [{"role": "user", "content": prompt}]
    call_chatgpt(message)


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
    call_chatgpt(message)


if __name__ == "__main__":
    # translate_question(5)
    demo_kt([3751, 3752, 3753, 3754, 1990, 3739, 3740, 3742, 3756, 3757], [1, 1, 1, 0, 1, 1, 1, 1, 0, 0])
