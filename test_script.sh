cd /opt/data/private/DFG

python3 test.py \
    --model_path '/opt/data/private/DFG/results/Target_Adapt/SAM_UNet_Abdomen_CT2MR/exp_0_time_2026-03-06 15:16:11/saved_models/best_model_step_5_dice_0.8604.pth' \
    --data_root datasets/Abdomen_Data_new \
    --target_site MRI \
    --gpu_id 0 \
    --save_vis \
    