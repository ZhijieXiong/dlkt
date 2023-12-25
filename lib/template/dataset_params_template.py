KT_RANDOM_AUG_PARAMS = {
    # 配置随机增强
    # 为True的话，在原序列基础上随机选一段做增强，如concept seq会产生concept_seq_ori,concept_seq_aug_0, concept_seq_aug_1, ...
    "random_select_aug_len": False,
    "mask_prob": 0.1,
    "replace_prob": 0.1,
    "crop_prob": 0.1,
    "permute_prob": 0.1,
    "hard_neg_prob": 1.0,
    "aug_order": ["mask", "replace", "permute", "crop"]
}


KT_INFORMATIVE_AUG_PARAMS = {
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


KT_LONG_TAIL_PARAMS = {
    "low_threshold": 0.2,
    "high_threshold": 0.8,

}
