import argparse
import os
import numpy as np

from lib.data_processor.load_raw import load_csv
from lib.util.parse import str2bool
from lib.util.data import write_json

from keys import jiujiuai_api_key
from llm_util import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\assist2012")
    parser.add_argument("--multi_concept", type=str2bool, default=False)
    parser.add_argument("--model_name", type=str, default="text-embedding-ada-002")

    args = parser.parse_args()
    params = vars(args)

    concept_id2name_map_path = os.path.join(params["data_dir"], "concept_id2name_map.csv")
    concept_id2name_map = load_csv(concept_id2name_map_path)

    concept_embedding_path = os.path.join(params["data_dir"], "concept_embeddings.json")
    question_embedding_path = os.path.join(params["data_dir"], "question_embeddings.json")
    if params["multi_concept"]:
        Q_table_path = os.path.join(params["data_dir"], "Q_table_multi_concept.npy")
        Q_table = np.load(Q_table_path)
    else:
        texts = get_text4single_concept(concept_id2name_map)
        texts_index = []
        concept_texts = []
        question_texts = []
        for k, v in texts.items():
            texts_index.append(k)
            concept_texts.append(v["concept"])
            question_texts.append(v["question"])

        concept_text_embeddings = []
        question_text_embeddings = []
        num_concept = len(texts)
        for i in range(num_concept // 10 + 1):
            concept_text_embeddings += call_chatgpt_embedding(jiujiuai_api_key, "text-embedding-ada-002", concept_texts[10 * i: 10 * (i+1)])
            question_text_embeddings += call_chatgpt_embedding(jiujiuai_api_key, "text-embedding-ada-002", question_texts[10 * i: 10 * (i+1)])

        concept_embeddings = {c_id: emb for c_id, emb in
                              zip(texts_index, [e["embedding"] for e in concept_text_embeddings])}
        question_embeddings = {c_id: emb for c_id, emb in
                               zip(texts_index, [e["embedding"] for e in question_text_embeddings])}
        write_json(concept_embeddings, concept_embedding_path)
        write_json(question_embeddings, question_embedding_path)



