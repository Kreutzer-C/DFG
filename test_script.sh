cd /workspace/DFG

python3 test.py \
    --model_path '/workspace/DFG/results/Target_Adapt/SAM_UNet_Abdomen_CT2MR_SAM/exp_0_time_2026-04-07 15:12:55/saved_models/best_model_step_5_dice_0.8718.pth' \
    --data_root datasets/chaos \
    --target_site MR \
    --gpu_id 0 \
    --save_vis \
    