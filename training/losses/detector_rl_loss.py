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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

# ==============================================================================
#  3. Reward Calculation Module
# ==============================================================================
def unproject_points(coords_2d, depth, intrinsics):
    B, N, _ = coords_2d.shape
    fx = intrinsics[:, 0, 0].view(B, 1, 1); fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1); cy = intrinsics[:, 1, 2].view(B, 1, 1)
    u = torch.clamp(coords_2d[:, :, 0], 0, depth.shape[2] - 1).long()
    v = torch.clamp(coords_2d[:, :, 1], 0, depth.shape[1] - 1).long()
    batch_indices = torch.arange(B, device=coords_2d.device).view(B, 1).expand(-1, N)
    
    # [Modified] Directly return sampled point depth values for subsequent validity checks
    sampled_depth = depth[batch_indices, v, u].unsqueeze(-1)
    
    x = (coords_2d[:, :, 0:1] - cx) * sampled_depth / fx
    y = (coords_2d[:, :, 1:2] - cy) * sampled_depth / fy
    return torch.cat([x, y, sampled_depth], dim=2), sampled_depth.squeeze(-1)

def project_points(points_3d, pose, intrinsics):
    """
    Project 3D spatial points back to 2D image coordinates
    Opposite to unprojection, first convert 3D points to target camera coordinate system via pose matrix,
    then project to 2D coordinates using intrinsic parameters, returning projected depth z.
    """
    B, N, _ = points_3d.shape
    points_3d_homo = F.pad(points_3d, (0, 1), 'constant', 1.0)
    points_in_tgt_cam = torch.bmm(points_3d_homo, pose.transpose(1, 2))
    x, y, z = points_in_tgt_cam[:, :, 0], points_in_tgt_cam[:, :, 1], points_in_tgt_cam[:, :, 2]
    z = torch.clamp(z, min=1e-6)
    fx = intrinsics[:, 0, 0].view(B, 1); fy = intrinsics[:, 1, 1].view(B, 1)
    cx = intrinsics[:, 0, 2].view(B, 1); cy = intrinsics[:, 1, 2].view(B, 1)
    u = (fx * x / z) + cx; v = (fy * y / z) + cy
    return torch.stack([u, v], dim=2), z


# ==============================================================================
#  7. Grid-based Sampling Function
# ==============================================================================
def grid_based_sampling(logits_map, log_policy_map, grid_cell_size=8, n_points_per_grid=1, 
                          sampling_strategy='hybrid', num_global_samples=100):
    B, _, H, W = logits_map.shape
    device = logits_map.device
    
    all_indices = []

    # --- Part 1: Grid Sampling ---
    if sampling_strategy in ['grid', 'hybrid']:
        grid_h = H // grid_cell_size
        grid_w = W // grid_cell_size
        num_grids = grid_h * grid_w

        logits_grids = logits_map.unfold(2, grid_cell_size, grid_cell_size).unfold(3, grid_cell_size, grid_cell_size)
        logits_grids = logits_grids.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, num_grids, -1)
        
        chooser = Categorical(logits=logits_grids)
        local_indices = chooser.sample((n_points_per_grid,)).squeeze(0)

        grid_indices_y = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
        grid_indices_x = torch.arange(grid_w, device=device).repeat(grid_h)
        local_coords_y = torch.div(local_indices, grid_cell_size, rounding_mode='floor')
        local_coords_x = local_indices % grid_cell_size
        
        global_coords_y = grid_indices_y * grid_cell_size + local_coords_y
        global_coords_x = grid_indices_x * grid_cell_size + local_coords_x
        
        indices_grid = global_coords_y * W + global_coords_x
        all_indices.append(indices_grid)
   
    # --- Part 2: Global Sampling ---
    if sampling_strategy in ['global', 'hybrid']:
        # [Core Fix] Directly use logits for sampling to ensure numerical stability
        logits_flat = logits_map.view(B, -1)
        dist_global = Categorical(logits=logits_flat)
        indices_global = dist_global.sample((num_global_samples,)).transpose(0, 1)
        all_indices.append(indices_global)
    
    # --- Part 3: Combination ---
    final_indices = torch.cat(all_indices, dim=1)
    
    # Convert indices back to coordinates (for reward function)
    final_coords_y = torch.div(final_indices, W, rounding_mode='floor')
    final_coords_x = final_indices % W
    final_coords = torch.stack([final_coords_x, final_coords_y], dim=2)

    # [Core] All points' log_probs are looked up from the global policy map
    log_policy_flat = log_policy_map.view(B, -1)
    log_probs = torch.gather(log_policy_flat, 1, final_indices)
    return final_coords.float(), log_probs



def calculate_advanced_rewardv2(ref_sampled_coords,  images, depths, intrinsics, rel_poses_to_ref, all_desc_maps, all_logits, grid_cell_size,
                              depth_thresh, match_thresh, w_track, acceptance_mask, current_warm_weight, ratio_thresh , seq_len):
    depth_penalty = 0.0
    desc_penalty = 0.0
    ratio_test_reward = 0.5
    
    rank_warmup_thresh = 0.5
    B, N_KEYPOINTS, _ = ref_sampled_coords.shape
    device = ref_sampled_coords.device
    _, _, H_img, W_img = depths.shape
    ref_desc_map = all_desc_maps[:, 0]
    norm_coords = ref_sampled_coords.clone()
    norm_coords[:, :, 0] = 2 * (norm_coords[:, :, 0] / (W_img - 1)) - 1
    norm_coords[:, :, 1] = 2 * (norm_coords[:, :, 1] / (H_img - 1)) - 1
    ref_descs = F.grid_sample(ref_desc_map, norm_coords.unsqueeze(1), mode='bilinear', align_corners=True).squeeze(2).permute(0, 2, 1)
    
    # [New] accumulated_rewards is used to accumulate reward/penalty scores for each frame
    accumulated_rewards = torch.zeros(B, N_KEYPOINTS, device=device)
    covisibility_count = torch.zeros(B, N_KEYPOINTS, device=device)
    
    # [Core Modification] First check reference frame depth validity
    points_3d_ref, ref_depths = unproject_points(ref_sampled_coords, depths[:, 0], intrinsics[:, 0])
    mask_valid_ref_depth = ref_depths > 0

    for j in range(seq_len - 1):
     
        projected_coords, projected_depth = project_points(points_3d_ref, rel_poses_to_ref[:, j], intrinsics[:, j+1])
        mask_boundary = (projected_coords[..., 0] >= 0) & (projected_coords[..., 0] < W_img) & \
                        (projected_coords[..., 1] >= 0) & (projected_coords[..., 1] < H_img)

        # [Core Modification] Only points with valid reference depth can be considered as having co-visibility opportunities
        covisibility_mask_this_frame = mask_boundary
        covisibility_count[covisibility_mask_this_frame] += 1
        
        actual_depth = torch.zeros_like(projected_depth)
        u_proj, v_proj = projected_coords[..., 0].long(), projected_coords[..., 1].long()
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(-1, N_KEYPOINTS)
        actual_depth[mask_boundary] = depths[:, j+1][batch_indices[mask_boundary], v_proj[mask_boundary], u_proj[mask_boundary]]
        mask_valid_actual_depth = actual_depth > 0
        
        # [Core Modification] Implement the more refined depth consistency check you designed
        # 1. By default, all points pass depth consistency check
        mask_depth_consistency = torch.ones_like(mask_valid_ref_depth).bool()
        # 2. Find points where both reference and target frame depths are valid
        both_depths_valid_mask = mask_valid_ref_depth & mask_valid_actual_depth & mask_boundary
        # 3. Only calculate consistency error for these points
        if both_depths_valid_mask.any():
            consistency_error = (torch.abs(projected_depth[both_depths_valid_mask] - actual_depth[both_depths_valid_mask]) / torch.clamp(actual_depth[both_depths_valid_mask], 1e-6)) < depth_thresh
            # 4. Update the calculated error results to the final mask
            mask_depth_consistency[both_depths_valid_mask] = consistency_error
        
        target_desc_map = all_desc_maps[:, j+1]
        norm_proj_coords = projected_coords.clone()
        norm_proj_coords[:, :, 0] = 2 * (norm_proj_coords[:, :, 0] / (W_img - 1)) - 1
        norm_proj_coords[:, :, 1] = 2 * (norm_proj_coords[:, :, 1] / (H_img - 1)) - 1
        target_descs = F.grid_sample(target_desc_map, norm_proj_coords.unsqueeze(1), mode='bilinear', align_corners=True).squeeze(2).permute(0, 2, 1)
        
        # [Modified] Perform explicit L2 normalization before calculating similarity to ensure cosine similarity computation
        ref_descs_norm = F.normalize(ref_descs, p=2, dim=-1)
        target_descs_norm = F.normalize(target_descs, p=2, dim=-1)
        
        similarities = (ref_descs_norm * target_descs_norm).sum(dim=-1)
        mask_desc_match = similarities > match_thresh


        # [Core Fix] Calculate ranking directly in Logits space for clearer and more efficient logic
        target_logits_map = all_logits[:, j+1]
        # 1. Use bilinear interpolation to get precise logit values for projected points
        proj_logits = F.grid_sample(target_logits_map, norm_proj_coords.unsqueeze(1), mode='bilinear', align_corners=True).squeeze(1).squeeze(1)

        # 2. Get all logit values from the grid where the projected point is located
        grid_h, grid_w = H_img // grid_cell_size, W_img // grid_cell_size
        logits_grids = target_logits_map.unfold(2, grid_cell_size, grid_cell_size).unfold(3, grid_cell_size, grid_cell_size)
        logits_grids = logits_grids.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, grid_h * grid_w, -1)
        
        clamped_proj_x = torch.clamp(projected_coords[..., 0], 0, W_img - 1)
        clamped_proj_y = torch.clamp(projected_coords[..., 1], 0, H_img - 1)
        proj_grid_y = torch.div(clamped_proj_y, grid_cell_size, rounding_mode='floor').long()
        proj_grid_x = torch.div(clamped_proj_x, grid_cell_size, rounding_mode='floor').long()
        proj_grid_idx = proj_grid_y * grid_w + proj_grid_x
        
        gathered_logit_grids = torch.gather(logits_grids, 1, proj_grid_idx.unsqueeze(-1).expand(-1, -1, logits_grids.shape[-1]))
        
        # 3. Directly compare logit values to calculate ranking
        rank = (gathered_logit_grids < proj_logits.unsqueeze(-1)).sum(dim=-1)
        
        # 4. Convert ranking to normalized reward
        num_points_in_grid = grid_cell_size * grid_cell_size
        rank_reward = rank.float() / num_points_in_grid

        # --- Hierarchical Reward/Penalty Application (Vectorized) ---
        current_frame_reward = torch.zeros(B, N_KEYPOINTS, device=device)
        depth_fail_mask = covisibility_mask_this_frame & ~mask_depth_consistency
        current_frame_reward[depth_fail_mask] = depth_penalty
        desc_fail_mask = covisibility_mask_this_frame & mask_depth_consistency & ~mask_desc_match
        current_frame_reward[desc_fail_mask] = desc_penalty
        rank_check_mask = covisibility_mask_this_frame & mask_depth_consistency & mask_desc_match
        
       
        use_ratio_test = True
        # current_warm_weight (0.1-1)
        rank_reward_thresh = 0.8 # default: 0.8
        rank_penalty_thresh = 0.5
        ratio_reward_thresh = 0.85 # default: 0.85
        ratio_penalty_thresh = 0.91
        
       
        if rank_check_mask.any():
            
            rank_pass_mask = rank_reward >= (1.0 - rank_reward_thresh)
            rank_fail_mask = rank_reward < rank_penalty_thresh
            
            scaled_reward = (rank_reward - (1.0 - rank_reward_thresh)) / rank_reward_thresh
            scaled_penalty = (rank_reward - rank_penalty_thresh) / rank_penalty_thresh
            scaled_reward = torch.clamp(scaled_reward, 0.0, 1.0)
            scaled_penalty = torch.clamp(scaled_penalty, -0.2, 0.0)
            
            rank_score = torch.zeros_like(rank_reward)
            rank_score[rank_pass_mask] = scaled_reward[rank_pass_mask]

            
            # 2. Calculate Ratio score ([-1, 1] range)
            ratio_score = torch.zeros_like(rank_reward)
            if use_ratio_test:
                dists = torch.cdist(ref_descs_norm, target_descs_norm, p=2)
                nn_dists = dists.diagonal(dim1=-2, dim2=-1).clone()
                dists.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
                snn_dists, _ = torch.min(dists, dim=-1)
                ratio = nn_dists / (snn_dists + 1e-8)
                
                pass_mask_ratio = ratio < ratio_reward_thresh
                fail_mask_ratio = ratio >= ratio_penalty_thresh
                
                scaled_reward_ratio = (ratio_reward_thresh - ratio) / ratio_reward_thresh
                scaled_reward_ratio = torch.clamp(scaled_reward_ratio, 0.0, 1.0)

                ratio_score[pass_mask_ratio] = scaled_reward_ratio[pass_mask_ratio]

            # 4. Accumulate final score
            current_frame_reward[rank_check_mask] += rank_score[rank_check_mask] 
            if use_ratio_test:
                current_frame_reward[rank_check_mask] += ratio_score[rank_check_mask] 
            
        accumulated_rewards += current_frame_reward
       
    per_point_success_rate = accumulated_rewards / torch.clamp(covisibility_count, min=1)
    # Create a mask to select only points that have at least one co-visibility opportunity
    covisible_mask = covisibility_count > 0
    
    # Set non-co-visible points' success rate to 0 so they don't participate in summation
    masked_success_rate = per_point_success_rate * covisible_mask.float()

    # Calculate the number of co-visible points for each sample
    num_covisible_points = covisible_mask.sum(dim=1)

    # After summation, divide by the number of co-visible points to get average success rate
    # clamp(min=1) is to avoid division by zero
    R_track = masked_success_rate.sum(dim=1) / torch.clamp(num_covisible_points, min=1)
    
    total_reward = w_track * R_track 

    return total_reward, covisibility_count, mask_valid_ref_depth

def spatial_entropy_regularization(policy_map, grid_size=16):
    
    B, _, H, W = policy_map.shape
   
    cell_h = H // grid_size
    cell_w = W // grid_size
    
   
    if H % grid_size != 0 or W % grid_size != 0:
       
        policy_grids = policy_map.unfold(2, cell_h, cell_h).unfold(3, cell_w, cell_w)
        policy_grids = policy_grids[:, :, :grid_size, :grid_size, :, :] 
    else:
        policy_grids = policy_map.view(B, 1, grid_size, cell_h, grid_size, cell_w)

    policy_grids = policy_grids.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, grid_size * grid_size, -1)
    
   
    policy_grid_norm = policy_grids / (policy_grids.sum(dim=2, keepdim=True) + 1e-8)
    
   
    entropy_per_grid = - (policy_grid_norm * torch.log(policy_grid_norm + 1e-8)).sum(dim=2)
  
    return entropy_per_grid.mean()


class ReinforceLoss(nn.Module):
    def __init__(self, lambda_entropy: float = 0.01):
        super().__init__(); self.lambda_entropy = lambda_entropy
        
    def forward(self, policy_map, log_probs, avg_reward, valid_mask, lambda_entropy):
        log_probs_sum = log_probs.sum(dim=1) # Total log probability for each sample
        loss_rl = - (avg_reward.detach() * log_probs_sum).mean() # Reward-guided update (detach to avoid reward participating in gradient calculation)
        # [Core Modification] Call the new spatial entropy function
        entropy = spatial_entropy_regularization(policy_map, grid_size=16)
        
        total_loss = loss_rl - lambda_entropy * entropy

        loss_components = {"total_loss": total_loss, "rl_loss": loss_rl.item(),
                           "entropy": entropy, "avg_reward": avg_reward.mean().item()}
        return loss_components

