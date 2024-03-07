MODEL_PARAMS = {
    "encoder_layer": {
        "GIKT": {
            "num_concept": 123,
            "num_question": 17751,
            "dim_emb": 64,
            "num_q_neighbor": 4,
            "num_c_neighbor": 10,
            "agg_hops": 2,
            "hard_recap": True,
            "rank_k": 10,
            "dropout4gru": 0.3,
            "dropout4gnn": 0.4
        }
    }
}
