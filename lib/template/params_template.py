PARAMS = {
  "device": "cuda",
  "seed": 0,
  "save_model": False,
  "save_model_dir": "",
  # 数据预处理部分配置
  "preprocess_config": {
    "dataset_name": "assist2009",
    "data_path": ""
  },
  # 训练策略部分配置
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
  # 其它loss（joint loss）的权重配置
  "loss_config": {
    # "rasch_loss": 0.00001
    # "cl_loss": 0.1
  },
  # 模型参数配置
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
        },
        "AKT4cold_start": {
          "num_concept": 123,
          "num_question": 17751,
          "dim_model": 64,
          "key_query_same": True,
          "num_head": 8,
          "num_block": 2,
          "dim_ff": 128,
          "dim_final_fc": 256,
          "dropout": 0.3,
          "separate_qa": False,
          "cold_start_step1": 5,
          "cold_start_step2": 10,
          "effect_start_step2": 0.5
        },
        "AT_DKT": {
          "num_concept": 123,
          "num_question": 17751,
          "dim_emb": 64,
          "dim_latent": 64,
          "rnn_type": "lstm",
          "num_rnn_layer": 1,
          # "transformer" or "rnn"
          "QT_net_type": "transformer",
          "QT_rnn_type": "lstm",
          "QT_num_rnn_layer": 1,
          "QT_transformer_num_block": 2,
          "QT_transformer_num_head": 8,
          "dropout": 0.3,
          "IK_start": 50
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
  # 优化器配置
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
    #   "share": True,
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
  # 学习率衰减配置
  "schedulers_config": {
    "kt_model": {
      "use_scheduler": False,
      # 例如"StepLR"
      "type": "StepLR",
      "StepLR": {
        "step_size": 10,
        "gamma": 0.1
      },
      "MultiStepLR": {
        "milestones": [5, 10],
        "gamma": 0.1
      }
    },
    # "other_model1": {
    #   "share": True,
    #   "use_scheduler": False,
    #   "type": "StepLR",
    #   "StepLR": {
    #     "step_size": 10,
    #     "gamma": 0.1
    #   },
    # }
  },
  # 学习率裁剪配置
  "grad_clip_config": {
    "kt_model": {
      "use_clip": False,
      "grad_clipped": 10.0
    },
    # "other_model1": {
    #   "share": True,
    # }
  },
  # 数据集配置（训练集、测试集、验证集）
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
        # "random_aug" or "semantic_aug" or "informative_aug"
        "aug_type": "semantic_aug",
        "num_aug": 2,
        "random_aug": {
          # 配置随机增强
          # 为True的话，在原序列基础上随机选一段做增强，如concept seq会产生concept_seq_ori,concept_seq_aug_0, concept_seq_aug_1, ...
          "random_select_aug_len": False,
          "mask_prob": 0.1,
          "replace_prob": 0.1,
          "crop_prob": 0.1,
          "permute_prob": 0.1,
          "hard_neg_prob": 1.0,
          "aug_order": ["mask", "replace", "permute", "crop"]
        },
        "informative_aug": {
          # 配置info增强
          "random_select_aug_len": False,
          "mask_prob": 0.1,
          "insert_prob": 0.1,
          "replace_prob": 0.3,
          "crop_prob": 0.1,
          # "order" or ""
          "offline_sim_type": "order",
          "num_concept": 123,
          "num_question": 17751,
          # "offline" or "online" or "hybrid"
          "sim_type": "off",
          "aug_order": ["mask", "crop", "replace", "insert"]
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
  # 其它参数配置，如对比学习中warm up设置
  "other": {
    "duo_cl": {
      "temp": 0.05,
      "cl_type": "last_time",
    },
    "instance_cl": {
      "temp": 0.05,
      "use_warm_up4cl": False,
      "epoch_warm_up4cl": 4,
      "use_online_sim": True,
      "use_warm_up4online_sim": True,
      "epoch_warm_up4online_sim": 4,
      # "last_time" or "all_time" or "mean_pool"
      "cl_type": "last_time",
      "use_adv_data": False
    },
    "max_entropy_aug": {
      "use_adv_aug": False,
      "epoch_interval_generate": 1,
      "loop_adv": 3,
      "epoch_generate": 40,
      "adv_learning_rate": 10,
      "eta": 5,
      "gamma": 1
    },
    "cluster_cl": {
      "random_select_aug_len": False,
      "num_cluster": 32,
      "temp": 0.05,
      "use_warm_up4cl": False,
      "epoch_warm_up4cl": 4,
      "use_online_sim": True,
      "use_warm_up4online_sim": True,
      "epoch_warm_up4online_sim": 4,
      "cl_type": "last_time",
      "use_adv_data": False
    },
    "max_entropy_adv_aug": {
      "use_warm_up": True,
      "epoch_warm_up": 4,
      "epoch_interval_generate": 1,
      "loop_adv": 3,
      "epoch_generate": 40,
      "adv_learning_rate": 10,
      "eta": 5,
      "gamma": 1
    }
  }
}
