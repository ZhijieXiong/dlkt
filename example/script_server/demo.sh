#!/bin/bash

{
  lrs=("0.001" "0.0005" "0.0001")
  # lrGammaRows和lrStepRows的每一行对应lrs的每一个元素
  lrGammaRows=(
    "0.1 0.25 0.5"
    "0.25 0.5"
    "0.5"
  )
  lrStepRows=(
    "5 10 15"
    "10 15"
    "10 20"
  )
  # lrDecNumRows和lrGammaRows一一对应
  lrDecNumRows=(
    "3 4 4"
    "4 4"
    "4"
  )

  for ((i=0; i<${#lrs[@]}; i++)); do
    lr=${lrs[$i]}
    lrSteps=(${lrStepRows[$i]})
    lrGammas=(${lrGammaRows[$i]})
    lrDecNums=(${lrDecNumRows[$i]})
    for ((j=0; j<${#lrGammas[@]}; j++)); do
      lrDecNum=${lrDecNums[$j]}
      lrGamma=${lrGammas[$j]}
      for lrStep in "${lrSteps[@]}"; do
        numEpoch=$((lrStep*lrDecNum))

        echo -e "lr: ${lr}, gamma: ${lrGamma}, step: ${lrStep}"
        python /ghome/xiongzj/code/pykt-dream/example/task/our/final.py \
          --model_name "qDKT" --useful_info "{'seq_len', 'concept_seq', 'correct_seq', 'mask_seq', 'question_seq'}" \
          --setting_name "domain_school-leave_multi_out-setting-6" --dataset_name "assist2009" \
          --seed 0 --model_dir_prefix "" \
          --save_model False \
          --learning_rate "${lr}" --batch_size 64 --enable_clip_grad False \
          --enable_lr_schedule True --lr_schedule_step "${lrStep}" --lr_schedule_gamma "${lrGamma}" \
          --num_epoch ${numEpoch} --last_epoch 5 \
          --dim_emb 64 --dropout 0.1 \
          --use_cl True --temp 0.05 --use_projector False \
          --warm_up4cl True --warm_up_epoch4cl 4 --warm_up4info_aug True --warm_up_epoch4info_aug 4 \
          --use_neg_generated False --type_neg_generated "MoCo" --num_neg_generated 512 \
          --use_instance_cl True --coef_instance_cl_loss 0.1 \
          --use_intent_cl False --num_cluster 32 --coef_intent_cl_loss 0.3 \
          --use_random_aug False --views4random 0 --mask4random_aug 0.1 --crop4random_aug 0.1 --permute4random_aug 0.1 --replace4random_aug 0.3 --negative4random_aug 1.0 \
          --use_info_aug True --views4info 2 --insert4info_aug 0.2 --mask4info_aug 0 --replace4info_aug 0.3 --crop4info_aug 0.1 \
          --use_adv_aug False --views4adv 0 --epoch_inter_gen 1 --loop_adv 5 --num_generate 100 --adv_learning_rate 20 --eta 1.0 --gamma 1.0

      done
    done
  done
} >> /ghome/xiongzj/code/pykt-dream/result/cluster/our/observation/instanceCL_qdkt_LMO_as09_domain_6.txt