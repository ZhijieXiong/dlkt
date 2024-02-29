MODEL_PARAMS = {
    "kt_embed_layer": {
        # 都是[num_emb, dim_emb]的格式，除了use LLM emb
        "concept": [],
        "question": [],
        "interaction": [],
        "use_LLM_emb": False
    },
    "encoder_layer": {
        "AT_DKT": {
            "dim_concept": 64,
            "dim_question": 64,
            "dim_correct": 64,
            "dim_latent": 64,
            "rnn_type": "gru",
            "num_rnn_layer": 1
        },
    }
}
