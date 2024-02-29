MODEL_PARAMS = {
    "encoder_layer": {
        "SimpleKT": {
            "num_concept": 123,
            "num_question": 17751,
            "dim_model": 64,
            "num_head": 8,
            "num_block": 2,
            "dim_ff": 128,
            "dim_final_fc": 256,
            "dim_final_fc2": 256,
            "dropout": 0.3,
            "seq_len": 200,
            "key_query_same": True,
            "separate_qa": False,
            "difficulty_scalar": True
        }
    }
}