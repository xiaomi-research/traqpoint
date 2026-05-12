import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from typing import Optional, Callable
from ..utils.misc import NestedTensor

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=False)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))
        x = self.gate(self.bn2(self.conv2(x)))
        return x

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=False)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.gate(out)

        return out

class TraqPoint_detector(nn.Module):
    def __init__(self, block_dims, hidden_dim=64):
        super().__init__()
        self.gate = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.block1 = ConvBlock(3, block_dims[0], self.gate, nn.BatchNorm2d)
        self.block2 = ResBlock(inplanes=block_dims[0], planes=block_dims[1], stride=1,
                            downsample=nn.Conv2d(block_dims[0], block_dims[1], 1),
                            gate=self.gate,
                            norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(inplanes=block_dims[1], planes=block_dims[2], stride=1,
                            downsample=nn.Conv2d(block_dims[1], block_dims[2], 1),
                            gate=self.gate,
                            norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(inplanes=block_dims[2], planes=block_dims[3], stride=1,
                            downsample=nn.Conv2d(block_dims[2], block_dims[3], 1),
                            gate=self.gate,
                            norm_layer=nn.BatchNorm2d)

        self.conv1 = resnet.conv1x1(block_dims[0], hidden_dim // 4)
        self.conv2 = resnet.conv1x1(block_dims[1], hidden_dim // 4)
        self.conv3 = resnet.conv1x1(block_dims[2], hidden_dim // 4)
        self.conv4 = resnet.conv1x1(block_dims[3], hidden_dim // 4)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.policy_head = resnet.conv1x1(hidden_dim, 1)

    def _forward_logits(self, samples: NestedTensor) -> torch.Tensor:
        x1 = self.block1(samples.tensors)
        x2 = self.pool2(x1)
        x2 = self.block2(x2)
        x3 = self.pool4(x2)
        x3 = self.block3(x3)
        x4 = self.pool4(x3)
        x4 = self.block4(x4)

        x1 = self.gate(self.conv1(x1))
        x2 = self.gate(self.conv2(x2))
        x3 = self.gate(self.conv3(x3))
        x4 = self.gate(self.conv4(x4))

        x2_up = self.upsample2(x2)
        x3_up = self.upsample8(x3)
        x4_up = self.upsample32(x4)

        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        return self.policy_head(x1234)

    def forward(self, samples: NestedTensor):
        logits = self._forward_logits(samples)

        if not self.training:
            B, C, H, W = logits.shape
            logits_flat = logits.view(B, -1)
            max_vals = logits_flat.max(dim=1, keepdim=True)[0]
            min_vals = logits_flat.min(dim=1, keepdim=True)[0]
            scale = max_vals - min_vals
            scale = torch.clamp(scale, min=1e-6)
            norm_logits_flat = (logits_flat - min_vals) / scale
            policy_map = norm_logits_flat.view_as(logits)
            log_policy_map = torch.log(policy_map.clamp(min=1e-8))
            logits = torch.sigmoid(logits)
        else:
            logits_flat = logits.view(logits.size(0), -1)
            stable_logits_flat = logits_flat - logits_flat.max(dim=1, keepdim=True)[0]
            log_policy_flat = F.log_softmax(stable_logits_flat, dim=1)
            log_policy_map = log_policy_flat.view_as(logits)
            policy_flat = F.softmax(stable_logits_flat, dim=1)
            policy_map = policy_flat.view_as(logits)
        return policy_map, log_policy_map, logits

    def forward_with_raw(self, samples: NestedTensor):
        """RDD-style policy/log_policy computation (softmax/log_softmax on logits).

        This method is intended for pose-only inference paths (e.g. mega_1500/scannet_1500)
        without affecting the default eval-time behavior used by other downstream tasks.
        """
        logits = self._forward_logits(samples)
        logits_flat = logits.view(logits.size(0), -1)
        stable_logits_flat = logits_flat - logits_flat.max(dim=1, keepdim=True)[0]
        log_policy_flat = F.log_softmax(stable_logits_flat, dim=1)
        log_policy_map = log_policy_flat.view_as(logits)
        policy_flat = F.softmax(stable_logits_flat, dim=1)
        policy_map = policy_flat.view_as(logits)
        return policy_map, log_policy_map, logits

def build_detector(config):
    block_dims = [8, 16, 32, 64]
    return TraqPoint_detector(block_dims, block_dims[-1])
