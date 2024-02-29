MODEL_PARAMS = {
    "kt_embed_layer": {
        # 都是[num_emb, dim_emb]的格式，除了use LLM emb
        "concept": [],
        "question": [],
        "use_LLM_emb": True
    },
    "encoder_layer": {
        "qDKT": {
            "dim_concept": 64,
            "dim_question": 64,
            "dim_correct": 64,
            "dim_latent": 64,
            "rnn_type": "gru",
            "num_rnn_layer": 1
        },
    },
    "predict_layer": {
        # "direct" or "product"
        "type": "direct",
        "direct": {
          "dropout": 0.3,
          "num_predict_layer": 1,
          "dim_predict_in": 64,
          "dim_predict_mid": 128,
          "dim_predict_out": 123,
          "activate_type": "sigmoid"
        },
        "product": {

        }
    }
}
