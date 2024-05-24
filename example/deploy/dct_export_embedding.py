import json
import sys
import os
import inspect
import torch


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
sys.path.append(settings["LIB_PATH"])

from lib.util.FileManager import FileManager
from lib.model.DCT import DCT
from lib.util.data import load_json, read_preprocessed_file, write_json
from lib.util.parse import question2concept_from_Q, concept2question_from_Q
from lib.model.Module.KTEmbedLayer import KTEmbedLayer


if __name__ == "__main__":
    save_model_dir = r"F:\code\myProjects\dlkt\lab\saved_models\2024-05-21@19-25-59@@DCT@@seed_0@@baidu_competition@@train_dataset"
    its_question_path = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition\its_question.json"
    save_model_name = "saved.ckt"
    ckt_name = "saved.ckt"
    model_name_in_ckt = "best_valid"
    dataset_name = "xes3g5m"
    data_type = "only_question"
    device = "cuda"
    test_file_path = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition\test_data.txt"
    target_dir = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition"
    file_manager = FileManager(FILE_MANAGER_ROOT)

    its_question = load_json(its_question_path)

    global_objects = {}
    Q_table = file_manager.get_q_table(dataset_name, data_type)
    global_objects["data"] = {}
    global_objects["data"]["Q_table"] = Q_table
    if Q_table is not None:
        global_objects["data"]["Q_table_tensor"] = torch.from_numpy(Q_table).long().to(device)
        global_objects["data"]["question2concept"] = question2concept_from_Q(Q_table)
        global_objects["data"]["concept2question"] = concept2question_from_Q(Q_table)
        q2c_table, q2c_mask_table, num_max_concept = KTEmbedLayer.parse_Q_table(Q_table, device)
        global_objects["data"]["q2c_table"] = q2c_table
        global_objects["data"]["q2c_mask_table"] = q2c_mask_table
        global_objects["data"]["num_max_concept"] = num_max_concept

    params_path = os.path.join(save_model_dir, "params.json")
    global_params = load_json(params_path)
    global_params["device"] = device

    ckt_path = os.path.join(save_model_dir, ckt_name)
    model = DCT(global_params, global_objects).to(device)
    saved_ckt = torch.load(ckt_path)
    model.load_state_dict(saved_ckt[model_name_in_ckt])

    data_test = read_preprocessed_file(test_file_path)
    user_portrait = {}
    question_kt_portrait = []

    model.eval()
    with torch.no_grad():
        question_emb = model.get_question_emb_list()
        for q_id, q_emb in enumerate(question_emb):
            question = {
                "question_id": its_question[str(q_id)]["question_id"],
                "question_emb": list(map(lambda x: round(x, 10), q_emb))
            }
            question_kt_portrait.append(question)

        batch_size = 8
        batch_idx = [i for i in range(0, len(data_test) - batch_size, batch_size)]
        for i_start in batch_idx:
            batch_data = data_test[batch_size * i_start: batch_size * i_start + batch_size]
            if len(batch_data) == 0:
                break
            user_names = [f"xes3g5mUser{item_data['user_id']}" for item_data in batch_data]
            batch = model.get_user_batch(batch_data)
            batch_size = batch["mask_seq"].shape[0]
            first_index = torch.arange(batch_size).long().to(device)
            user_latent = model.get_user_latent(batch)[first_index, batch["seq_len"] - 1].detach().cpu().numpy().tolist()
            user_ability = model.get_user_ability(batch)[first_index, batch["seq_len"] - 1].detach().cpu().numpy().tolist()

            for i, user_name in enumerate(user_names):
                user_portrait[user_name] = {
                    "ability": user_ability[i],
                    "latent": user_latent[i]
                }

    write_json(user_portrait, os.path.join(target_dir, "its_user_portrait.json"))
    write_json(question_kt_portrait, os.path.join(target_dir, "its_question_kt_portrait.json"))
