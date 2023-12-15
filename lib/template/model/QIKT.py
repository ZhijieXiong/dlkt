MODEL_PARAMS = {
    "kt_embed_layer": {
        # 都是[num_emb, dim_emb]的格式
        "concept": [],
        "question": [],
        "correct": [],
        "interaction": []
    },
    "encoder_layer": {
        "QIKT": {
            "num_concept": 123,
            "num_question": 17751,
            "dim_emb": 64,
            "rnn_type": "gru",
            "num_rnn_layer": 1,
            "num_mlp_layer": 2,
            "dropout": 0.4,
            "lambda_q_all": 1,
            "lambda_c_next": 1,
            "lambda_c_all": 1,
            "use_irt": True
        },
    }
}
