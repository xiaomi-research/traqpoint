# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    import MultiScaleDeformableAttention as MSDA
except ModuleNotFoundError as e:
    info_string = (
        "\nThe MultiScaleDeformableAttention CUDA extension is not compiled. "
        "Using the slower PyTorch implementation instead.\n"
        "\tTo compile it, run the following commands:\n"
        "\tcd TraqPoint/models/ops\n"
        "\tpip install -e ."
    )
    warnings.warn(info_string)


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Multi-scale deformable attention in pure PyTorch (for CPU fallback and testing)."""
    batch, spatial_len, num_heads, channels = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    split_sizes = [int(h * w) for h, w in value_spatial_shapes]
    value_per_level = value.split(split_sizes, dim=1)

    # Convert sampling locations from [0, 1] to grid_sample's [-1, 1] range
    grids = 2 * sampling_locations - 1

    sampled_values = []
    for level_idx, (h, w) in enumerate(value_spatial_shapes):
        # (batch, h*w, num_heads, channels) -> (batch*num_heads, channels, h, w)
        val = value_per_level[level_idx].flatten(2).transpose(1, 2).reshape(
            batch * num_heads, channels, int(h), int(w))
        # (batch, num_queries, num_heads, num_points, 2) -> (batch*num_heads, num_queries, num_points, 2)
        grid = grids[:, :, :, level_idx].transpose(1, 2).flatten(0, 1)
        # (batch*num_heads, channels, num_queries, num_points)
        sampled = F.grid_sample(val, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_values.append(sampled)

    # (batch, num_queries, num_heads, num_levels, num_points) -> (batch*num_heads, 1, num_queries, num_levels*num_points)
    weights = attention_weights.transpose(1, 2).reshape(
        batch * num_heads, 1, num_queries, num_levels * num_points)
    # Stack sampled values across levels, weight and sum
    output = (torch.stack(sampled_values, dim=-2).flatten(-2) * weights).sum(-1).view(
        batch, num_heads * channels, num_queries)
    return output.transpose(1, 2).contiguous()
