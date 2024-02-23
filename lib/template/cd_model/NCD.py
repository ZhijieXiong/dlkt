MODEL_PARAMS = {
    "backbone": {
        "type": "NCD",
        "NCD": {
            "num_user": 2500,
            "num_question": 17751,
            "num_concept": 123
        },
    },
    "predict_layer": {
        "dim_predict1": 256,
        "dim_predict2": 512,
        "dropout": 0.5
    }
}
