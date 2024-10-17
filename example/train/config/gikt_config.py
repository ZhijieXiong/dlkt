from ._config import *


def gen_gikt_graph(question2concept, concept2question, q_neighbor_size, c_neighbor_size):
    num_question = len(question2concept)
    num_concept = len(concept2question)
    q_neighbors = np.zeros([num_question, q_neighbor_size], dtype=np.int32)
    c_neighbors = np.zeros([num_concept, c_neighbor_size], dtype=np.int32)
    for q_id, neighbors in question2concept.items():
        if len(neighbors) >= q_neighbor_size:
            q_neighbors[q_id] = np.random.choice(neighbors, q_neighbor_size, replace=False)
        else:
            q_neighbors[q_id] = np.random.choice(neighbors, q_neighbor_size, replace=True)
    for c_id, neighbors in concept2question.items():
        if len(neighbors) >= c_neighbor_size:
            c_neighbors[c_id] = np.random.choice(neighbors, c_neighbor_size, replace=False)
        else:
            c_neighbors[c_id] = np.random.choice(neighbors, c_neighbor_size, replace=True)
    return q_neighbors, c_neighbors


def get_bipartite_graph(question2concept, concept2question, device, q_neighbor_size, c_neighbor_size):
    num_question, num_concept = len(question2concept), len(concept2question)
    edge_index_q2c = [[], []]
    edge_attr_q2c = []
    edge_index_c2q = [[], []]
    edge_attr_c2q = []
    for q_index in range(num_question):
        neighbors = question2concept[q_index]
        if len(neighbors) == 0:
            continue
        if len(neighbors) < q_neighbor_size:
            neighbors = np.random.choice(neighbors, q_neighbor_size, replace=True)
        else:
            neighbors = np.random.choice(neighbors, q_neighbor_size)
        for c_neighbor in neighbors:
            edge_index_q2c[0].append(q_index + num_concept)
            edge_index_q2c[1].append(c_neighbor)
            edge_attr_q2c.append(1)
    for c_index in range(num_concept):
        neighbors = concept2question[c_index]
        if len(neighbors) == 0:
            continue
        if len(neighbors) < q_neighbor_size:
            neighbors = np.random.choice(neighbors, c_neighbor_size, replace=True)
        else:
            neighbors = np.random.choice(neighbors, c_neighbor_size)
        for q_neighbor in neighbors:
            edge_index_c2q[0].append(c_index)
            edge_index_c2q[1].append(q_neighbor + num_concept)
            edge_attr_c2q.append(1)
    edge_index_q2c = torch.tensor(edge_index_q2c, dtype=torch.int64).to(device)
    edge_attr_q2c = torch.tensor(edge_attr_q2c, dtype=torch.float).to(device)
    edge_index_c2q = torch.tensor(edge_index_c2q, dtype=torch.int64).to(device)
    edge_attr_c2q = torch.tensor(edge_attr_c2q, dtype=torch.float).to(device)
    return edge_index_q2c, edge_attr_q2c, edge_index_c2q, edge_attr_c2q


def gikt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "GIKT",
                "GIKT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    num_q_neighbor = local_params["num_q_neighbor"]
    num_c_neighbor = local_params["num_c_neighbor"]
    agg_hops = local_params["agg_hops"]
    hard_recap = local_params["hard_recap"]
    rank_k = local_params["rank_k"]
    dropout4gru = local_params["dropout4gru"]
    dropout4gnn = local_params["dropout4gnn"]

    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["GIKT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_emb"] = dim_emb
    encoder_config["num_q_neighbor"] = num_q_neighbor
    encoder_config["num_c_neighbor"] = num_c_neighbor
    encoder_config["agg_hops"] = agg_hops
    encoder_config["hard_recap"] = hard_recap
    encoder_config["rank_k"] = rank_k
    encoder_config["dropout4gru"] = dropout4gru
    encoder_config["dropout4gnn"] = dropout4gnn

    global_objects["logger"].info(
        "model params\n"
        f"    num_concept: {num_concept}, num_question: {num_question}\n    dim_emb: {dim_emb}, num_q_neighbor: {num_q_neighbor}, "
        f"num_c_neighbor: {num_c_neighbor}, agg_hops: {agg_hops}, hard_recap: {hard_recap}, rank_k: {rank_k}, "
        f"dropout4gru: {dropout4gru}, dropout4gnn: {dropout4gnn}"
    )

    # 配置需要的数据
    global_objects["gikt"] = {}
    if local_params["use_pyg"]:
        edge_index_q2c, edge_attr_q2c, edge_index_c2q, edge_attr_c2q = get_bipartite_graph(
            global_objects["data"]["question2concept"],
            global_objects["data"]["concept2question"],
            global_params["device"],
            num_q_neighbor,
            num_c_neighbor
        )
        global_objects["gikt"]["edge_index"] = (edge_index_q2c, edge_index_c2q)
        global_objects["gikt"]["edge_attr"] = (edge_attr_q2c, edge_attr_c2q)
    else:
        q_neighbors, c_neighbors = gen_gikt_graph(
            global_objects["data"]["question2concept"],
            global_objects["data"]["concept2question"],
            num_q_neighbor,
            num_c_neighbor
        )
        global_objects["gikt"]["question_neighbors"] = q_neighbors
        global_objects["gikt"]["concept_neighbors"] = c_neighbors

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"GIKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def gikt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    gikt_general_config(local_params, global_params, global_objects)

    return global_params, global_objects
