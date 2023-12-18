import requests
import argparse
import os
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt

from keys import jiujiuai_api_key

from lib.data_processor.load_raw import load_csv


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt_embedding(model_name, text, max_tokens=8000):
    url = "https://www.jiujiuai.life/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": jiujiuai_api_key
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


def get_question_text(q_table, q_id):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/Users/dream/Desktop/code/projects/dlkt/lab/dataset_preprocessed/assist2012")
    parser.add_argument("--model_name", type=str, default="text-embedding-ada-002")

    args = parser.parse_args()
    params = vars(args)

    concept_id2name_map_path = os.path.join(params["data_dir"], "concept_id2name_map.csv")
    Q_table_path = os.path.join(params["data_dir"], "Q_table_single_concept.npy")
    concept_embedding_path = os.path.join(params["data_dir"], "concept_embeddings.json")
    question_embedding_path = os.path.join(params["data_dir"], "question_embeddings.json")

    concept_id2name_map = load_csv(concept_id2name_map_path)
    Q_table = np.load(Q_table_path)


