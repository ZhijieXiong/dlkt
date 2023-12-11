from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests

from lib.util.data import load_json


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


if __name__ == "__main__":
    data_dir = "/Users/dream/Desktop/code/projects/dlkt/lab/dataset_preprocessed/xes3g5m"
    question_meta = load_json(f"{data_dir}/question_meta.json")
    concept_meta = load_json(f"{data_dir}/concept_meta.json")
    question_seq = [3751, 3752, 3753, 3754, 1990, 3739, 3740, 3742, 3756, 3757]
    question_seq = list(map(str, question_seq))
    question_content_seq = list(map(lambda x: question_meta[x]["content"], question_seq))
    concept_content_seq = list(map(
        lambda x: "; ".join(question_meta[x]["kc_routes"]),
        question_seq
    ))
    correct_seq = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
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

    message = [{"role": "user", "content": prompt}]
    # 注意max_tokens不能太大
    question, n_tokens = call_chatgpt(message, model="gpt-3.5-turbo", max_tokens=10000)
    print(question)
