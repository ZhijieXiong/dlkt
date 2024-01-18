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


def demo_kt(ask_question_seq, ask_correct_seq, next_q_id, use_question_content=False):
    prompt_name = f"KT_prompt_{'w' if use_question_content else 'wo'}_text.md"
    example_prompt_all = load_prompt(prompt_name)

    ask_question_seq = list(map(str, ask_question_seq))
    ask_question_content_seq = list(map(lambda x: QUESTION_CONTENT[x]["content_only_text"], ask_question_seq))
    ask_concept_content_seq = list(map(
        lambda x: "; ".join(QUESTION_CONTENT[x]["kc_routes"]),
        ask_question_seq
    ))
    ask_correct_seq = list(map(lambda x: "做对" if x else "做错", ask_correct_seq))
    ask_seq_len = len(ask_question_seq)
    qa_prompt = f"Student SA’s practice record is "
    if use_question_content:
        qa_prompt += f"[`{', '.join([f'(`{ask_question_content_seq[i]}`, `{ask_concept_content_seq[i]}`, `{ask_correct_seq[i]}`)' for i in range(ask_seq_len)])}]"
    else:
        qa_prompt += f"[`{', '.join([f'(`{ask_concept_content_seq[i]}`, `{ask_correct_seq[i]}`)' for i in range(ask_seq_len)])}]"
    qa_prompt += f", \n\nPlease determine whether SA can do this exercise correctly: {QUESTION_CONTENT[str(next_q_id)]['content_only_text']}\n"

    prompt = f"{example_prompt_all}\n" \
             f"input: {qa_prompt}\n" \
             f"output: \n"

    message = [{"role": "user", "content": prompt}]
    call_chatgpt(message)


def demo_question_rec(question_seq_, correct_seq_, use_question_content=False):
    prompt_name = f"ExerciseRec_prompt.md"
    example_prompt = load_prompt(prompt_name)

    prompt = f"{example_prompt}\n"

    message = [{"role": "user", "content": prompt}]
    call_chatgpt(message)


def translate_question(question_id):
    question_content = QUESTION_CONTENT[str(question_id)]
    prompt_name = f"Translation_prompt_{'fill_up' if question_content['type'] == '填空' else 'choice'}.md"
    example_prompt = load_prompt(prompt_name)

    prompt = f"{example_prompt}\n" \
             f"input: \n\n" \
             f"{question_content['content_with_image_name']}\n" \
             f"output: \n"

    message = [{"role": "user", "content": prompt}]
    call_chatgpt(message)


def load_prompt(prompt_name):
    prompt_path = "./{}".format(prompt_name)
    with open(prompt_path, 'r', encoding='utf-8') as fp:
        prompt = fp.read().strip() + "\n\n"
    return prompt


def check_question_seq(q_seq, correct_seq_):
    q_seq = list(map(str, q_seq))
    q_content_seq = list(map(lambda x: QUESTION_CONTENT[x]["content_only_text"], q_seq))
    c_content_seq = list(map(
        lambda x: "; ".join(QUESTION_CONTENT[x]["kc_routes"]),
        q_seq
    ))
    correct_seq_ = list(map(lambda x: "做对" if x else "做错", correct_seq_))

    for q, c, correct_ in zip(q_content_seq, c_content_seq, correct_seq_):
        print(f"question: {q}\n"
              f"concept: {c}\n"
              f"result: {correct_}")


def ask_java_question():
    pass


if __name__ == "__main__":
    # translate chinese to english
    # translate_question(20)

    # KT
    # question_seq = [3751, 3752, 3753, 3754, 1990, 3739, 3740, 3742, 3756]
    # correct_seq = [1, 1, 1, 0, 1, 1, 1, 1, 0]
    # assert len(question_seq) == len(correct_seq), "len of question_seq must be equal to len of correct_seq"
    # next_q = 3757
    # next_correct = 0
    # demo_kt(question_seq, correct_seq, next_q, use_question_content=False)

    # check
    question_seq = [266, 268, 2203, 870, 2316, 2587, 2323, 2588, 5194]
    correct_seq = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    check_question_seq(question_seq, correct_seq)
    next_qs = [868, 869, 2206]
    next_corrects = [0, 0, 0]
