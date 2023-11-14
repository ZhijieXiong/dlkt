PARAMS = {
  "device": "cuda",
  "seed": 0,
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
    "use_multi_metrics": False,
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
    # "rasch_loss": 0.00001
    # "cl_loss": 0.1
  },
  "models_config": {
    "kt_model": {
      "kt_embed_layer": {
        # 都是[num_emb, dim_emb]的格式
        "concept": [],
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
        },
        "qDKT": {
          "dim_concept": 64,
          "dim_question": 64,
          "dim_correct": 64,
          "dim_latent": 64,
          "rnn_type": "gru",
          "num_rnn_layer": 1
        },
        "AKT": {
          "num_concept": 123,
          "num_question": 17751,
          "dim_model": 64,
          "key_query_same": True,
          "num_head": 8,
          "num_block": 2,
          "dim_ff": 128,
          "dim_final_fc": 256,
          "dropout": 0.3,
          "separate_qa": False
        }
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
    "data_type": "multi_concept",
    "train": {
      # "kt" or "kt4aug" or "srs"
      "type": "kt",
      "setting_name": "pykt_setting",
      "file_name": "assist2009_train_fold_0.txt",
      "unuseful_seq_keys": {"user_id"},
      "kt": {
        # 配置KTDataset需要的参数
        "base_type": "concept"
      },
      "kt4aug": {
        # "random_aug" or "semantic_aug"
        "aug_type": "semantic_aug",
        "num_aug": 2,
        "random_aug": {
          # 配置随机增强
          "mask_prob": 0.1,
          "replace_prob": 0.1,
          "crop_prob": 0.1,
          "permute_prob": 0.1,
          "hard_neg_prob": 1.0,
          "aug_order": ["mask", "replace", "permute", "crop"]
        },
        "informative_aug": {
          # 配置info增强
          "mask_prob": 0.1,
          "replace_prob": 0.1,
          "crop_prob": 0.1,
          "offline_sim_type": "order"
        }
      }
    },
    "valid": {
      "type": "kt",
      "setting_name": "pykt_setting",
      "file_name": "assist2009_valid_fold_0.txt",
      "unuseful_seq_keys": {"user_id"},
      "kt": {
        "base_type": "concept"
      },
    },
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
  "other": {
    "duo": {
      "temp": 0.05
    },
    "informative_aug_config": {
      "num_concept": 123,
      "num_question": 17751,
    }
  }
}
