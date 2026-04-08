cd /workspace/DFG

python3 test.py \
    --model_path '/workspace/DFG/results/Target_Adapt/SAM_UNet_Abdomen_MR2CT_SAM/exp_0_time_2026-04-08 07:25:24/saved_models/best_model_step_10_dice_0.7574.pth' \
    --data_root datasets/chaos \
    --target_site CT \
    --gpu_id 0 \
    --save_vis \
    