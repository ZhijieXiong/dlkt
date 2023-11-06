params = {
  "file_manager_root": "",
  "device": "cuda",
  "preprocess_config": {
    "dataset_name": "assist2009",
    "data_path": ""
  },
  "train_strategy": {
    "type": "valid_test / no_valid",
    "valid_test": {
      "use_early_stop": True,
      "epoch_early_stop": 10,
      "main_metric": "AUC / ACC / MAE / RMSE",
      "use_mutil_metric": False,
      "multi_metrics": [("AUC", 1), ("ACC", 1)]
    },
    "no_valid": {
      "use_average": True,
      "epoch_last_average": 5
    }
  },
  "models_config": {
    "kt_model": {
      "kt_embed_layer": {
        "concept": [123, 64],
        "question": [17000, 64],
        "correct": [2, 128],
        "interaction": [246, 64]
      },
      "encoder_layer": {
        "type": "AKT",
        "params": []
      },
      "predict_layer": ""
    }
  },
  "optimizers_config": {
    "kt_mode": {
      "type": "adam / sgd",
      "adam": {
        "lr": "0.001",
        "weight_decay": "0.0",
      },
      "sgd": {
        "lr": "0.01",
        "weight_decay": "0.999",
        "momentum": "0.9"
      }
    },
    # "other_model1": {
    #   "type": "adam / sgd",
    #   "adam": {
    #     "lr": "0.001",
    #     "weight_decay": "0.0",
    #   },
    #   "sgd": {
    #     "lr": "0.01",
    #     "weight_decay": "0.999",
    #     "momentum": "0.9"
    #   }
    # }
  },
  "schedulers_config": {
    "kt_model": {
      "type": "StepLR",
      "StepLR": {
        "step_size": 10,
        "gamma": 0.1
      },
    },
    # "other_model1": {
    #   "type": "StepLR",
    #   "StepLR": {
    #     "step_size": 10,
    #     "gamma": 0.1
    #   },
    # }
  },
  "datasets_config": {
    "type": "kt / srs",
    "setting_name": "",
    "dataset_name": "",
    "use_aug": False,
    "aug_config": {
      "type": "random / info"
    },
    "train_path": "",
    "valid_path": "",
    "test_path": ""
  },
  # 配置KTDataset需要的参数
  "dataset_this_config": {
    "dataset_path": "",
    "data_type": "multi_concept",
    "unuseful_seq_keys": {"user_id"},
    "base_type": "concept"
  }
}