from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, functional
from timm.layers import DropPath

from amzcls.models.builder import BACKBONES
from amzcls.neurons import build_node, NODES
from amzcls.neurons.layer import LayerNorm
from ..utils.timm import to_2tuple, trunc_normal_, _cfg

__all__ = ['SpikformerImageNet', 'spikformer_in1k']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, neuron_cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = layer.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = layer.BatchNorm2d(hidden_features)
        self.fc1_lif = build_node(neuron_cfg)

        self.fc2_conv = layer.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = layer.BatchNorm2d(out_features)
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
        self.scale = 0.125
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

    def forward(self, x, res_attn):
        T, B, C, H, W = x.shape
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        x = (q @ (k.transpose(-2, -1) @ v)) * self.scale
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W))
        return x, v


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., norm_layer=LayerNorm, neuron_cfg=None):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads, neuron_cfg=neuron_cfg)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, neuron_cfg=neuron_cfg)

    def forward(self, x, res_attn):
        x_attn, attn = self.attn(x, res_attn)
        x = x + x_attn
        x = x + (self.mlp(x))
        return x, attn


class SPS(nn.Module):
    def __init__(self, img_size_h=224, img_size_w=224, patch_size=4, in_channels=3, embed_dims=256,
                 neuron_cfg=None):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = layer.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = layer.BatchNorm2d(embed_dims // 8)
        self.proj_lif = build_node(neuron_cfg)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = layer.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = layer.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = build_node(neuron_cfg)
        self.maxpool1 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = layer.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = layer.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = build_node(neuron_cfg)
        self.maxpool2 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = layer.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = layer.BatchNorm2d(embed_dims)
        self.proj_lif3 = build_node(neuron_cfg)
        self.maxpool3 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = layer.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = layer.BatchNorm2d(embed_dims)
        self.rpe_lif = build_node(neuron_cfg)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x)
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

        x_feat = x
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)
        x = x + x_feat
        return x


default_neuron = dict(type='LIFNode', tau=2.0, detach_reset=True)


@BACKBONES.register_module()
class SpikformerImageNet(nn.Module):
    def __init__(
            self, img_size_h=224, img_size_w=224, patch_size=16, in_channels=3,
            embed_dims=512, num_heads=8, mlp_ratios=4, drop_path_rate=0.2,
            norm_layer=partial(LayerNorm, eps=1e-6), depths=8, neuron_cfg=None
    ):
        super().__init__()
        if neuron_cfg is None:
            print(f"[INFO] Using default neuron `{default_neuron}`.\n"
                  "\tfrom `amzcls.models.backbones.spikformer_dvs`.")
            neuron_cfg = default_neuron
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims,
                          neuron_cfg=neuron_cfg)

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, drop_path=dpr[j],
            norm_layer=norm_layer, neuron_cfg=neuron_cfg)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
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
        attn = None
        for blk in block:
            x, attn = blk(x, attn)
        return x

    def forward(self, x):
        functional.reset_net(self)
        x = self.forward_features(x)
        return x,


@BACKBONES.register_module()
def spikformer_in1k(
        img_size_h=224, img_size_w=224, patch_size=16, in_channels=3, num_classes=1000, embed_dims=512,
        num_heads=8, mlp_ratios=4, drop_path_rate=0.2, norm_layer=partial(LayerNorm, eps=1e-6), depths=8):
    model = SpikformerImageNet(
        img_size_h, img_size_w, patch_size, in_channels, num_classes, embed_dims,
        num_heads, mlp_ratios, drop_path_rate, norm_layer, depths,
    )
    model.default_cfg = _cfg()
    return model
