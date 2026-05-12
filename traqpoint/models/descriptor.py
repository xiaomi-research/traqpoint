import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.misc import NestedTensor, nested_tensor_from_tensor_list
import torchvision.transforms as transforms
from .backbone_dinov3_conv import build_backbone
from .deformable_transformer import build_deforamble_transformer

class BasicLayer(nn.Module):
    """Conv2d -> BatchNorm -> ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.layer(x)


class TraqPoint_Descriptor(nn.Module):
    def __init__(self, backbone, transformer, num_feature_levels):
        super().__init__()
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels

        self.matchibility_head = nn.Sequential(
            BasicLayer(256, 128, 1, padding=0),
            BasicLayer(128, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(768, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.backbone = backbone
        self.stride = backbone.strides[0]
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        flatten_feats, spatial_shapes, level_start_index = self.transformer(srcs, masks, pos)

        feats = []
        level_start_index = torch.cat((level_start_index, torch.tensor([flatten_feats.shape[1] + 1]).to(level_start_index.device)))
        for i, shape in enumerate(spatial_shapes):
            assert len(shape) == 2
            temp = flatten_feats[:, level_start_index[i]: level_start_index[i + 1], :]
            feats.append(temp.transpose(1, 2).view(-1, self.hidden_dim, *shape))

        final_feature = feats[0]
        for feat in feats[1:]:
            final_feature = final_feature + F.interpolate(feat, size=final_feature.shape[-2:], mode='bilinear', align_corners=True)

        matchibility = self.matchibility_head(final_feature)

        return final_feature, matchibility


def build_descriptor(config):
    backbone = build_backbone(config)
    transformer = build_deforamble_transformer(config)
    return TraqPoint_Descriptor(backbone, transformer, config['num_feature_levels'])
