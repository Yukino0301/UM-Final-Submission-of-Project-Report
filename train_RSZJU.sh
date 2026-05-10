#!/usr/bin/env bash

SEQUENCES=("my_377" "my_386" "my_387" "my_392" "my_393" "my_394")

MODEL_RS_NUM=11
DATA_RS_NUM=11

for SEQUENCE in "${SEQUENCES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        -s ./data/BlurZJU/rs${DATA_RS_NUM}/${SEQUENCE} \
        --motion_offset_flag --smpl_type smpl --actor_gender neutral \
        --exp_name RSZJU/drs${DATA_RS_NUM}/${SEQUENCE} \
        --port 6010 \
        --model_blur_num ${MODEL_RS_NUM} \
        --data_blur_num ${DATA_RS_NUM} \
        --render_mode rolling_shutter \
        --scan_direction top_to_bottom \
        --pose_mode spline_4 \
        --use_pose_offset \
        --use_lbs_offset \
        --resolution 2 \
        --iterations 40000 \
        --eval_every 5000 \
        --max_sh 3 \
        --pose_loss 1.0 \
        --ssim_weight 0.01 \
        --lpips_weight 0.01
done
