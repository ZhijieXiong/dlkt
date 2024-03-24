MODEL_PARAMS = {
    "kt_embed_layer": {
        "concept": [],
        "question": [],
        "correct": [],
        "use_LLM_emb": True
    },
    "encoder_layer": {
        "type": "AuxInfoQDKT",
        "AuxInfoQDKT": {
            "dim_question": 64,
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
          "activate_type": "relu"
        },
        "product": {

        }
    }
}
