MODEL_PARAMS = {
    "encoder_layer": {
        "NCD4KT": {
            "num_question": 17751,
            "num_concept": 123,
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
          "dim_predict_in": 123,
          "dim_predict_mid": 128,
          "dim_predict_out": 1,
          "activate_type": "sigmoid"
        },
        "product": {

        }
    }
}