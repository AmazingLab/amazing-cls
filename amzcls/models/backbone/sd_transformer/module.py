import torch.nn as nn
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.neuron import (
    LIFNode,
    ParametricLIFNode,
)

from ..utils.timm import to_2tuple


class MS_SPS(nn.Module):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=4,
            in_channels=2,
            embed_dims=256,
            pooling_stat="1111",
            spike_mode="lif",
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = layer.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = layer.BatchNorm2d(embed_dims // 8)
        if spike_mode == "lif":
            self.proj_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.proj_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool = layer.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = layer.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = layer.BatchNorm2d(embed_dims // 4)
        if spike_mode == "lif":
            self.proj_lif1 = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif1 = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool1 = layer.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = layer.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = layer.BatchNorm2d(embed_dims // 2)
        if spike_mode == "lif":
            self.proj_lif2 = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif2 = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool2 = layer.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = layer.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = layer.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.proj_lif3 = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif3 = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool3 = layer.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = layer.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = layer.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.rpe_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.rpe_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

    def forward(self, x):
        T, B, _, H, W = x.shape
        x = self.proj_conv(x)  # have some fire value
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = x
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)

        x = x
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)

        x = x
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)

        x_feat = x
        x = self.proj_lif3(x)

        x = x
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_Conv(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            spike_mode="lif",
            layer_index=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = layer.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = layer.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.fc2_conv = layer.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = layer.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer_index = layer_index

    def forward(self, x):
        identity = x
        x = self.fc1_lif(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = x + identity
        return x


class MS_SSA_Conv(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            mode="direct_xor",
            spike_mode="lif",
            dvs=False,
            layer_index=0,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_conv = layer.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = layer.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.q_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.q_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.k_conv = layer.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = layer.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.k_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.k_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.v_conv = layer.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = layer.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.v_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.v_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        if spike_mode == "lif":
            self.attn_lif = LIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.attn_lif = ParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.talking_heads = layer.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = LIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = ParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.proj_conv = layer.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = layer.BatchNorm2d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = LIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.shortcut_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.mode = mode
        self.layer_index = layer_index

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        identity = x
        x = self.shortcut_lif(x)
        x_for_qkv = x

        q_conv_out = self.q_lif(self.q_bn(self.q_conv(x_for_qkv)))
        q = q_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_lif(self.k_bn(self.k_conv(x_for_qkv)))
        k_conv_out = self.pool(k_conv_out) if self.dvs else k_conv_out
        k = k_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_lif(self.v_bn(self.v_conv(x_for_qkv)))
        v_conv_out = self.pool(v_conv_out) if self.dvs else v_conv_out
        v = v_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        kv = k.mul(v)
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        x = q.mul(kv)
        x = self.pool(x) if self.dvs else x
        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = self.proj_bn(self.proj_conv(x))

        x = x + identity
        return x, v


class MS_Block_Conv(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            attn_mode="direct_xor",
            spike_mode="lif",
            dvs=False,
            layer_index=0,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer_index=layer_index,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
            layer_index=layer_index,
        )

    def forward(self, x):
        x_attn, attn = self.attn(x)
        x = self.mlp(x_attn)
        return x, attn
