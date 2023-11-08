PARAMS = {
  "device": "cuda",
  "save_model": False,
  "preprocess_config": {
    "dataset_name": "assist2009",
    "data_path": ""
  },
  "train_strategy": {
    # "valid_test" or "no_valid"
    "type": "valid_test",
    "num_epoch": 200,
    # "AUC" or "ACC" or "MAE" or "RMSE"
    "main_metric": "AUC",
    "use_mutil_metrics": False,
    "multi_metrics": [("AUC", 1), ("ACC", 1)],
    "valid_test": {
      "use_early_stop": True,
      "epoch_early_stop": 10
    },
    "no_valid": {
      "use_average": True,
      "epoch_last_average": 5
    }
  },
  "loss_config": {
    # loss名称和权重，如对比损失cl loss
    "cl loss": 0.1
  },
  "models_config": {
    "kt_model": {
      "kt_embed_layer": {
        # 都是[num_emb, dim_emb]的格式
        "concept": [123, 64],
        "question": [],
        "correct": [],
        "interaction": []
      },
      "encoder_layer": {
        "type": "DKT",
        "DKT": {
          "dim_emb": 64,
          "dim_latent": 64,
          "rnn_type": "gru",
          "num_rnn_layer": 1,
        }
      },
      "predict_layer": {
        "type": "direct",
        "direct": {
          "dropout": 0.3,
          "num_predict_layer": 1,
          "dim_predict_mid": 128,
          "dim_predict_out": 123,
          "activate_type": "sigmoid"
        }
      }
    },
    # "other_model1": {}
  },
  "optimizers_config": {
    "kt_model": {
      # "adam" or "sgd"
      "type": "adam",
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
      "use_scheduler": False,
      # 例如"StepLR"
      "type": "StepLR",
      "StepLR": {
        "step_size": 10,
        "gamma": 0.1
      },
    },
    # "other_model1": {
    #   "use_scheduler": False,
    #   "type": "StepLR",
    #   "StepLR": {
    #     "step_size": 10,
    #     "gamma": 0.1
    #   },
    # }
  },
  "grad_clip_config": {
    "kt_model": {
      "use_clip": False,
      "grad_clipped": 10.0
    }
  },
  "datasets_config": {
    # 当前dataset的选择
    "dataset_this": "train",
    "train": {
      # 两种数据格式，"kt" or "srs"，后者是序列推荐的格式
      "type": "kt",
      "setting_name": "pykt_setting",
      "file_name": "assist2009_train_fold_0.txt",
      "batch_size": 64,
      "kt": {
        # 配置KTDataset需要的参数
        "data_type": "multi_concept",
        "unuseful_seq_keys": {"user_id"},
        "base_type": "concept"
      },
    },
    "valid": {
      "type": "kt",
      "setting_name": "pykt_setting",
      "file_name": "assist2009_valid_fold_0.txt",
      "batch_size": 64,
      "kt": {
        "data_type": "multi_concept",
        "unuseful_seq_keys": {"user_id"},
        "base_type": "concept"
      },
    },
    "test": {
      "type": "kt",
      "setting_name": "pykt_setting",
      "file_name": "assist2009_test.txt",
      "batch_size": 64,
      "kt": {
        "data_type": "multi_concept",
        "unuseful_seq_keys": {"user_id"},
        "base_type": "concept"
      },
    }
  }
}
