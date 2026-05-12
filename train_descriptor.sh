#!/bin/bash
# Stage-1 descriptor training entry (example). Adjust paths as needed.

python -m training.train_descriptor \
  --ckpt_save_path ./outputs/stage1-des \
  --distributed \
  --batch_size 32 \
  --training_res 800 \
  --lr 1e-4 \
  --gamma_steplr 0.5 \
  --save_ckpt_every 1000 \
  --test_every_iter 3000 \
  --epochs 12 
