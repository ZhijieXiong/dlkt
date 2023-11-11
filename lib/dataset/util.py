from ..util.parse import get_keys_from_uniform
from ..util.data import data_agg_question


def data_kt2srs(data_uniformed, data_type):
    if data_type == "multi_concept":
        data_uniformed = data_agg_question(data_uniformed)

    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    data_transformed = []
    for i, item_data in enumerate(data_uniformed):
        for j in range(1, item_data["seq_len"]):
            item_data_new = {
                "target_seq_id": i,
                "target_seq_len": j
            }
            for k in seq_keys:
                if k == "concept_seq":
                    item_data_new["target_concept"] = item_data[k][j]
                if k == "question_seq":
                    item_data_new["target_question"] = item_data[k][j]
                if k == "correct_seq":
                    item_data_new["target_correct"] = item_data[k][j]
                if k == "time_seq":
                    item_data_new["target_time"] = item_data[k][j]
                if k == "use_time_seq":
                    item_data_new["target_use_time"] = item_data[k][j]
            data_transformed.append(item_data_new)
    return data_transformed
