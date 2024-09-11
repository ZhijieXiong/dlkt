from load_idct import load_idct
from load_util import *


if __name__ == "__main__":
    data_path = r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_test_fold_0.txt"
    save_model_dir = r"F:\code\myProjects\dlkt\deploy\IDCT@@our_setting_new@@assist2009_train_fold_0@@seed_0@@2024-08-29@16-29-28"
    q_table_path = r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\assist2009\Q_table_multi_concept.npy"
    device = "cpu"

    data = read_preprocessed_file(data_path)
    model = load_idct(save_model_dir, device, q_table_path)
    q2c = get_global_objects_data(q_table_path, device)["question2concept"]

    # 输出json文件，包括每个学生历史练习习题，练习结果，练习过的知识点，最后一时刻的知识状态，最后一时刻最薄弱的知识点（练习过）
    result = {}
    target_path = r"F:\code\myProjects\dlkt\example4exercise_recommendation\data4idct\assist2009.json"
    model.eval()
    with torch.no_grad():
        for u_id, item_data in enumerate(data):
            if item_data["seq_len"] < 100:
                continue
            result[u_id] = {}
            batch = {
                "seq_len": torch.tensor([item_data["seq_len"]]).long().to(device),
                "question_seq": torch.tensor([item_data["question_seq"]]).long().to(device),
                "correct_seq": torch.tensor([item_data["correct_seq"]]).long().to(device),
                "mask_seq": torch.tensor([item_data["mask_seq"]]).long().to(device),
            }
            result[u_id]["ability"] = model.get_last_user_ability(batch).squeeze(0).detach().cpu().numpy().tolist()
            result[u_id]["ability"] = list(map(lambda x: round(x, 2), result[u_id]["ability"]))
            result[u_id]["question_seq"] = item_data["question_seq"][:item_data["seq_len"]]
            result[u_id]["correct_seq"] = item_data["correct_seq"][:item_data["seq_len"]]
            concept_history = []
            for q_id in result[u_id]["question_seq"]:
                concept_history += q2c[q_id]
            concept_history = sorted(list(set(concept_history)))
            concept_history_level = list(map(lambda x: result[u_id]["ability"][x], concept_history))
            concept_history_level = [(concept_history[i], concept_history_level[i]) for i in range(len(concept_history))]
            concept_history_sorted = sorted(concept_history_level, key=lambda x: x[1])
            result[u_id]["concept_history_sorted"] = concept_history_sorted
    write_json(result, target_path)
