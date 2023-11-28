PARAMS = {
  "device": "cuda",
  "seed": 0,
  "save_model": False,
  "save_model_dir": "",
  "save_model_name": "",
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
    "kt_model": None
    # "other_model1": None
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
    # "extractor": {
    #   "share_with_kt": False,
    #   "share_params_with_kt": True
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
    # "extractor": {
    #   "share_with_kt": False,
    #   "share_params_with_kt": True
    # }
  },
  # 学习率裁剪配置
  "grad_clip_config": {
    "kt_model": {
      "use_clip": False,
      "grad_clipped": 10.0
    },
    # "extractor": {
    #   "share_with_kt": False,
    #   "share_params_with_kt": True
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
        "aug_type": "",
        "num_aug": 2,
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
  "other": {}
}
