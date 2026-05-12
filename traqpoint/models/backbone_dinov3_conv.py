import torch
import os

import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple
from ..utils.misc import NestedTensor
from .position_encoding import PositionEmbeddingSine


def _require_path(path: str, what: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing required {what}: {path}\n"
            "This repo does not redistribute third-party code/weights. "
            "Run `bash scripts/setup_third_party.sh` and place required weight files as described in docs/THIRD_PARTY.md."
        )


class DINOV3ViTB16Backbone(nn.Module):
    def __init__(self, intermediate_layers=[0, 1, 2, 3]):
        super().__init__()

        self.patch_size = 16

        dinov3_repo_path = "./third_party/facebookresearch_dinov3_main"
        _require_path(dinov3_repo_path, "third-party repo (DINOv3)")


        self.dino = torch.hub.load(
            dinov3_repo_path,
            'dinov3_convnext_base',
            source='local',
            pretrained=False
        )
        self.checkpoint_path = "./third_party/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
        checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        self.dino.load_state_dict(checkpoint, strict=True)

        # Note: pretrained=True will download from the official URL if not cached.
        # In offline environments, place the weight file under the torch hub cache
        # (e.g., ~/.cache/torch/hub/checkpoints/) so this call can load it without network.
        # self.dino = torch.hub.load(
        #     './third_party/facebookresearch_dinov3_main',
        #     'dinov3_convnext_base',
        #     source='local',
        #     pretrained=True
        # )

        for param in self.dino.parameters():
            param.requires_grad_(False)

        assert hasattr(self.dino, 'stages') and len(self.dino.stages) == 4, "Backbone is not a ConvNeXt model"

        unfreeze_stages = [0, 1, 2, 3]
        for stage_idx in unfreeze_stages:
            current_stage = self.dino.stages[stage_idx]
            for param in current_stage.parameters():
                param.requires_grad = True

            current_norm = self.dino.norms[stage_idx]
            for param in current_norm.parameters():
                param.requires_grad = True

        self.intermediate_layers = intermediate_layers
        self.num_layers = len(intermediate_layers)
        self.num_channels = [128, 256, 512, 1024]
        self.strides = [4, 8, 16, 32]

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _get_dino_input_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """Compute input size padded to a multiple of the patch size (16)."""
        h, w = original_size
        dino_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        dino_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        return dino_h, dino_w

    def forward(self, tensor_list: NestedTensor):
        """Extract multi-scale features from DINOv3 ConvNeXt backbone."""
        imgs = tensor_list.tensors
        B, _, H, W = imgs.shape
        device = imgs.device

        dino_h, dino_w = self._get_dino_input_size((H, W))
        processed_imgs = self._preprocess_images(imgs, dino_h, dino_w).to(device)

        with torch.no_grad():
            all_feats = self.dino.get_intermediate_layers(
                processed_imgs,
                n=self.intermediate_layers
            )

        features = []

        stride_index = 0
        for feat in all_feats:
            target_h, target_w = H // self.strides[stride_index], W // self.strides[stride_index]
            feat = feat.reshape(B, target_h, target_w, feat.shape[2])
            feat = feat.permute(0, 3, 1, 2)
            stride_index = stride_index + 1
            features.append(feat)

        masks = self._generate_masks(tensor_list.mask, H, W)

        return [NestedTensor(feat, m) for feat, m in zip(features, masks)]

    def _preprocess_images(self, imgs, target_h, target_w):
        """Resize and normalize input images using tensor operations only."""
        resized_imgs = F.interpolate(
            imgs,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
        return (resized_imgs - mean) / std

    def _generate_masks(self, original_mask, orig_h, orig_w):
        """Downsample the input mask to match feature map resolutions."""
        masks = []

        stride_index = 0
        for _ in range(self.num_layers):
            target_h, target_w = orig_h // self.strides[stride_index], orig_w // self.strides[stride_index]
            mask_down = F.interpolate(
                original_mask.float().unsqueeze(1),
                size=(target_h, target_w),
                mode='bilinear'
            ).squeeze(1).bool()
            masks.append(mask_down)
            stride_index = stride_index + 1
        return masks


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        pos = [self[1](x).to(x.tensors.dtype) for x in xs]
        return xs, pos


def build_backbone(config):
    """Build the DINOv3 ConvNeXt feature extractor with positional encoding."""
    position_embedding = PositionEmbeddingSine(
        num_pos_feats=config['hidden_dim'] // 2,
        temperature=10000,
        normalize=True
    )

    backbone = DINOV3ViTB16Backbone(
        intermediate_layers=[0, 1, 2, 3]
    )

    model = Joiner(backbone, position_embedding)
    return model
