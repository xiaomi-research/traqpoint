python -m training.train_key --train_detector --distributed  --sampling_strategy hybrid --num_global_samples 112 \
--ckpt_save_path ./outputs/traqpoint_stage2_dino --grid_size 40 --ratio_thresh 0.5 --batch_size 120 --training_res 480 \
--weights ./weights/stage1_des.pth