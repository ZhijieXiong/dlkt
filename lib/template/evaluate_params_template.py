EVALUATE_PARAMS = {
    "save_model_dir": "",
    "device": "cpu",
    "datasets_config": {
        # 当前dataset的选择
        "dataset_this": "test",
        "data_type": "multi_concept",
        "test": {
            "type": "kt",
            "setting_name": "pykt_setting",
            "file_name": "assist2009_test.txt",
            "unuseful_seq_keys": {"user_id"},
            "kt": {
                "base_type": "concept"
            },
        }
    },
    "evaluate": {
        "fine_grain": {
            "max_seq_len": 200,
            "seq_len_absolute": [0, 5, 10, 20, 30, 50, 100, 150, 200],
            "statics_path": ""
        }
    }
}