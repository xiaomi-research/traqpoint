# Copyright (C) 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import torch
import torch.nn as nn


def get_expert_keypoints_and_heatmap(images_tensor, fast_threshold=20, nonmaxSuppression=True, gauss_sigma=2.0):
   
    B, C, H, W = images_tensor.shape
    device = images_tensor.device
    target_heatmaps = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=nonmaxSuppression)
    
    for i in range(B):
        img_tensor = images_tensor[i]
        img_np_chw = (img_tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        img_np_hwc = np.transpose(img_np_chw, (1, 2, 0))
        
        if C == 3:
            gray_image = cv2.cvtColor(img_np_hwc, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = img_np_hwc.squeeze(-1)

        keypoints = fast.detect(gray_image, None)
        
      
        temp_heatmap_np = np.zeros((H, W), dtype=np.float32)

        if keypoints:
            kpt_coords_x = [int(kp.pt[0]) for kp in keypoints]
            kpt_coords_y = [int(kp.pt[1]) for kp in keypoints]
            
            valid_indices_mask = (np.array(kpt_coords_y) < H) & (np.array(kpt_coords_x) < W)
            valid_y = np.array(kpt_coords_y)[valid_indices_mask]
            valid_x = np.array(kpt_coords_x)[valid_indices_mask]
            
            temp_heatmap_np[valid_y, valid_x] = 1.0

            
            kernel_size = int(6 * gauss_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            temp_heatmap_np = cv2.GaussianBlur(temp_heatmap_np, (kernel_size, kernel_size), gauss_sigma)
            
          
            if temp_heatmap_np.max() > 0:
                temp_heatmap_np /= temp_heatmap_np.max()
            
            target_heatmaps[i, 0] = torch.from_numpy(temp_heatmap_np).to(device)
            
    return target_heatmaps

class SupervisionLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.loss_fn = nn.MSELoss()

    def forward(self, logits, target_heatmap):
      
        predicted_heatmap = torch.sigmoid(logits)
        return self.loss_fn(predicted_heatmap, target_heatmap)
      