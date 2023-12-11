MODEL_PARAMS = {
    "kt_embed_layer": {
        # 都是[num_emb, dim_emb]的格式
        "concept": [],
        "question": [],
        "correct": [],
        "interaction": []
    },
    "rnn_layer": {
        "dim_concept": 64,
        "dim_question": 64,
        "dim_correct": 64,
        "dim_rnn": 100,
        "rnn_type": "gru",
        "num_rnn_layer": 1,
        "dropout": 0.2
    },
    "encoder_layer": {
        # "fc" or "fc_cnn" or "fc_no_res"
        "type": "fc",
        "dim_latent": 64,
        "add_eps": True
    }
}
