MODEL_PARAMS = {
    "kt_embed_layer": {
        "interaction": []
    },
    "encoder_layer": {
        "DKT": {
            "dim_emb": 64,
            "dim_latent": 64,
            "rnn_type": "gru",
            "num_rnn_layer": 1,
            "use_concept": True
        },
    },
    "predict_layer": {
        "direct": {
            "dropout": 0.3,
            "num_predict_layer": 1,
            "dim_predict_in": 64,
            "dim_predict_mid": 128,
            "dim_predict_out": 123,
            "activate_type": "sigmoid"
        }
    }
}
