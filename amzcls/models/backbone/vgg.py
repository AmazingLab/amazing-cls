import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from ..builder import MODELS
from ...neurons import build_node, NODES


class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, neuron_cfg):
        super(CBS, self).__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, (3, 3), padding=1)
        self.bn = layer.BatchNorm2d(out_channels)
        self.spike = build_node(neuron_cfg)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.spike(x)
        return x


def make_layers(in_channels, out_channels, num_layers, neuron_cfg):
    layers = [CBS(in_channels, out_channels, neuron_cfg)]
    for _ in range(num_layers - 1):
        cbs = CBS(out_channels, out_channels, neuron_cfg)
        layers.append(cbs)
        in_channels = in_channels

    return nn.Sequential(*layers)


default_neuron = dict(type='IFNode')
default_width = [64, 128, 256, 512]


@MODELS.register_module()
class VGG11(nn.Module):
    def __init__(self, layers: list, width: list = None,
                 in_channels=3, neuron_cfg=None):
        super(VGG11, self).__init__()
        if width is None:
            print(f"[INFO] Using default width `{default_width}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_resnet`.")
            width = default_width
        if neuron_cfg is None:
            print(f"[INFO] Using default neuron `{default_neuron}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_resnet`.")
            neuron_cfg = default_neuron
        self.dilation = 1
        self.layer1 = make_layers(in_channels, width[0], layers[0], neuron_cfg)
        self.mpool1 = layer.AvgPool2d(2, 2)
        self.layer2 = make_layers(width[0], width[1], layers[1], neuron_cfg)
        self.mpool2 = layer.AvgPool2d(2, 2)
        self.layer3 = make_layers(width[1], width[2], layers[2], neuron_cfg)
        self.mpool3 = layer.AvgPool2d(2, 2)
        self.layer4 = make_layers(width[2], width[3], layers[3], neuron_cfg)

        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend='cupy', instance=NODES.get(neuron_cfg['type']))

    def _forward_impl(self, x):
        functional.reset_net(self)
        x = self.layer1(x)
        x = self.mpool1(x)
        x = self.layer2(x)
        x = self.mpool2(x)
        x = self.layer3(x)
        x = self.mpool3(x)
        x = self.layer4(x)
        return x,

    def forward(self, x):
        return self._forward_impl(x)
