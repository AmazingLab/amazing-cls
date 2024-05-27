from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, functional

from amzcls.models.builder import BACKBONES
from amzcls.neurons import build_node, NODES
from amzcls.neurons.layer import LayerNorm, TimeEfficientBatchNorm2d
from ..utils.timm import to_2tuple, trunc_normal_, _cfg

__all__ = ['spikformer_dvs', 'SpikformerDVS']

default_neuron = dict(type='LIFNode', tau=2.0, detach_reset=True)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, neuron_cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = layer.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = layer.BatchNorm1d(hidden_features)
        self.fc1_lif = build_node(neuron_cfg)

        self.fc2_conv = layer.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = layer.BatchNorm1d(out_features)
        self.fc2_lif = build_node(neuron_cfg)
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        x = self.fc1_lif(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, neuron_cfg=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = layer.BatchNorm1d(dim)
        self.q_lif = build_node(neuron_cfg)

        self.k_conv = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = layer.BatchNorm1d(dim)
        self.k_lif = build_node(neuron_cfg)

        self.v_conv = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = layer.BatchNorm1d(dim)
        self.v_lif = build_node(neuron_cfg)

        self.attn_lif = build_node({**neuron_cfg, 'v_threshold': 0.5})
        self.proj_conv = layer.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = layer.BatchNorm1d(dim)
        self.proj_lif = build_node(neuron_cfg)

    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x
        q_conv_out = self.q_lif(self.q_bn(self.q_conv(x_for_qkv)))
        q = q_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_lif(self.k_bn(self.k_conv(x_for_qkv)))
        k = k_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_lif(self.v_bn(self.v_conv(x_for_qkv)))
        v = v_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.reshape(T, B, N, C).transpose(2, 3)
        x = self.attn_lif(x)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., norm_layer=LayerNorm, neuron_cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim, step_mode='m')
        self.attn = SSA(dim, num_heads=num_heads, neuron_cfg=neuron_cfg)
        self.norm2 = norm_layer(dim, step_mode='m')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, neuron_cfg=neuron_cfg)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256,
                 tdbn_step=None, neuron_cfg=None):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if tdbn_step is None:
            norm_layer = layer.BatchNorm2d
        else:
            norm_layer = partial(TimeEfficientBatchNorm2d, time_step=tdbn_step)
        self.proj_conv = layer.Conv2d(
            in_channels, embed_dims // 8, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.proj_bn = norm_layer(embed_dims // 8)
        self.proj_lif = build_node(neuron_cfg)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = layer.Conv2d(
            embed_dims // 8, embed_dims // 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.proj_bn1 = norm_layer(embed_dims // 4)
        self.proj_lif1 = build_node(neuron_cfg)
        self.maxpool1 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = layer.Conv2d(
            embed_dims // 4, embed_dims // 2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.proj_bn2 = norm_layer(embed_dims // 2)
        self.proj_lif2 = build_node(neuron_cfg)
        self.maxpool2 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = layer.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.proj_bn3 = norm_layer(embed_dims)
        self.proj_lif3 = build_node(neuron_cfg)
        self.maxpool3 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = layer.Conv2d(
            embed_dims, embed_dims, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.rpe_bn = norm_layer(embed_dims)
        self.rpe_lif = build_node(neuron_cfg)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x)  # have some fire value
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x))
        x_rpe = self.rpe_lif(x_rpe)
        x = x + x_rpe
        x = x.reshape(T, B, -1, (H // 16) * (H // 16)).contiguous()
        return x


@BACKBONES.register_module()
class SpikformerDVS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, embed_dims=256,
                 num_heads=16, mlp_ratios=4, norm_layer=None, tdbn_step=None, depths=2, neuron_cfg=None):
        super().__init__()
        if neuron_cfg is None:
            print(f"[INFO] Using default neuron `{default_neuron}`.\n"
                  "\tfrom `amzcls.models.backbones.spikformer_dvs`.")
            neuron_cfg = default_neuron

        self.depths = depths
        norm_layer = partial(LayerNorm, eps=1e-6) if norm_layer is None else norm_layer

        patch_embed = SPS(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims,
                          tdbn_step=tdbn_step,
                          neuron_cfg=neuron_cfg)
        pos_embed = nn.Parameter(torch.zeros(1, patch_embed.num_patches, embed_dims))

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, norm_layer=norm_layer, neuron_cfg=neuron_cfg)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend='cupy', instance=NODES.get(neuron_cfg['type']))

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x

    def forward(self, x):
        functional.reset_net(self)
        x = self.forward_features(x)
        return x.mean(3),


@BACKBONES.register_module()
def spikformer_dvs(norm_layer, **kwargs):
    model = SpikformerDVS(
        norm_layer=partial(LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
