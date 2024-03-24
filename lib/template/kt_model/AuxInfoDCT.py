MODEL_PARAMS = {
    "encoder_layer": {
        "type": "AuxInfoDCT",
        "AuxInfoDCT": {
            "num_question": 17751,
            "num_concept": 123,
            "dim_question": 64,
            "dim_latent": 64,
            "rnn_type": "gru",
            "num_rnn_layer": 1,
            "que_user_share_proj": False,
            "num_mlp_layer": 1,
            "dropout": 0.1
        },
    }
}