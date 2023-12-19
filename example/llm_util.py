import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt_embedding(api_key, model_name, text, max_tokens=8000):
    url = "https://www.jiujiuai.life/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }

    data = {
        "model": model_name,
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


def get_text4single_concept(concept_id2name_map):
    texts = {}
    for i, row in concept_id2name_map.iterrows():
        c_id = row["concept_id"]
        c_name = row["concept_name"].strip("\"")
        texts[c_id] = {
            "concept": f"This is a mathematical knowledge concept named `{c_name}`.",
            "question": f"This is a math question, and the knowledge concept examined is `{c_name}`."
        }

    return texts

