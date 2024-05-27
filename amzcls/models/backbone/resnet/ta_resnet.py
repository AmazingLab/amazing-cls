import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from .sew_resnet import SEWBottleneck, SEWBasicBlock, model_urls, conv1x1
from ...builder import MODELS, BACKBONES
from ....neurons import build_node, NODES

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = ['TAResNetCifar', 'ta_resnet18_cifar', 'ta_resnet34_cifar', 'ta_resnet50_cifar',
           'ta_resnet101_cifar', 'ta_resnet152_cifar', 'ta_resnext50_cifar_32x4d',
           'ta_resnext101_cifar_32x8d', 'ta_wide_resnet50_cifar_2', 'ta_wide_resnet101_cifar_2']


class TS(nn.Module):
    def __init__(self, rate):
        super(TS, self).__init__()
        self.rate = rate

    def forward(self, x):
        x = torch.reshape(x, (self.rate, x.shape[0] // self.rate, *x.shape[1:]))
        return x.mean(0)


class TSVec(nn.Module):
    def __init__(self, rate):
        super(TSVec, self).__init__()
        self.rate = rate
        self.weight = nn.Parameter(torch.randn((rate,)), requires_grad=True)

    def forward(self, x):
        x = torch.reshape(x, (self.rate, x.shape[0] // self.rate, *x.shape[1:]))
        x = torch.einsum('i,i...->...', self.weight, x)
        return x


default_neuron = dict(type='IFNode')
default_width = [64, 128, 256, 512]
default_stride = [1, 2, 2, 2]


@MODELS.register_module()
class TAResNetCifar(nn.Module):
    def __init__(self, block_type, layers: list, width: list = None, stride: list = None,
                 in_channels=3, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, cnf_list: tuple = ('add',), neuron_cfg=None):
        super().__init__()
        block = MODELS.get(block_type)
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if width is None:
            print(f"[INFO] Using default width `{default_width}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_resnet`.")
            width = default_width
        if stride is None:
            print(f"[INFO] Using default width `{default_stride}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_resnet`.")
            stride = default_stride
        if neuron_cfg is None:
            print(f"[INFO] Using default neuron `{default_neuron}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_resnet`.")
            neuron_cfg = default_neuron
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # image net: 3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        self.conv1 = layer.Conv2d(in_channels, self.inplanes, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = build_node(neuron_cfg)
        self.time_adaptive1 = TSVec(rate=2)
        self.time_adaptive2 = TSVec(rate=2)
        self.time_adaptive3 = TSVec(rate=2)
        self.layer1 = self._make_layer(block, width[0], layers[0], stride[0],
                                       cnf_list=cnf_list, neuron_cfg=neuron_cfg)
        self.layer2 = self._make_layer(block, width[1], layers[1], stride[1], dilate=replace_stride_with_dilation[0],
                                       cnf_list=cnf_list, neuron_cfg=neuron_cfg)
        self.layer3 = self._make_layer(block, width[2], layers[2], stride[2], dilate=replace_stride_with_dilation[1],
                                       cnf_list=cnf_list, neuron_cfg=neuron_cfg)
        self.layer4 = self._make_layer(block, width[3], layers[3], stride[3], dilate=replace_stride_with_dilation[2],
                                       cnf_list=cnf_list, neuron_cfg=neuron_cfg)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            print(f"[INFO] zero init residual: `{zero_init_residual}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_resnet`.")
            for m in self.modules():
                if isinstance(m, SEWBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, SEWBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend='cupy', instance=NODES.get(neuron_cfg['type']))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf_list: tuple = None, neuron_cfg=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        # cnf_list = ['or', 'iand']
        layers = [block(
            self.inplanes, planes, stride, downsample, groups=self.groups, base_width=self.base_width,
            dilation=self.dilation, norm_layer=norm_layer, cnf=cnf_list[0], neuron_cfg=neuron_cfg)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                dilation=self.dilation, norm_layer=norm_layer, cnf=cnf_list[i % len(cnf_list)], neuron_cfg=neuron_cfg))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        functional.reset_net(self)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.time_adaptive1(x)
        x = self.layer2(x)
        x = self.time_adaptive2(x)
        x = self.layer3(x)
        x = self.time_adaptive3(x)
        x = self.layer4(x)
        return x,

    def forward(self, x):
        return self._forward_impl(x)


def _resnet_cifar(arch, pretrained, progress, **kwargs):
    model = TAResNetCifar(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict.pop('conv1.weight')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


@BACKBONES.register_module()
def ta_resnet18_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet18', pretrained, progress, layers=[2, 2, 2, 2], **kwargs)


@BACKBONES.register_module()
def ta_resnet34_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet34', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def ta_resnet50_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet50', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def ta_resnet101_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet101', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def ta_resnet152_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet152', pretrained, progress, layers=[3, 8, 36, 3], **kwargs)


@BACKBONES.register_module()
def ta_resnext50_cifar_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet_cifar('resnext50_32x4d', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def ta_resnext101_cifar_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet_cifar('resnext101_32x8d', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def ta_wide_resnet50_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet50_2', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def ta_wide_resnet101_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet101_2', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)
