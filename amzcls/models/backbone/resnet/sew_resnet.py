import torch.nn as nn
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from spikingjelly.activation_based import layer, functional

from ...builder import MODELS, BACKBONES
from ....neurons import build_node, NODES

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = ['SEWResNetCifar', 'sew_resnet18_cifar', 'sew_resnet34_cifar', 'sew_resnet50_cifar',
           'sew_resnet101_cifar', 'sew_resnet152_cifar', 'sew_resnext50_cifar_32x4d',
           'sew_resnext101_cifar_32x8d', 'sew_wide_resnet50_cifar_2', 'sew_wide_resnet101_cifar_2',
           'SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50',
           'sew_resnet101', 'sew_resnet152', 'sew_resnext50_32x4d',
           'sew_resnext101_32x8d', 'sew_wide_resnet50_2', 'sew_wide_resnet101_2'
           ]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def sew_function(cnf: str):
    if cnf == 'add':
        return lambda x, y: x + y
    elif cnf == 'and':
        return lambda x, y: x * y
    elif cnf == 'iand':
        return lambda x, y: x * (1. - y)
    elif cnf == 'or':
        # 1. - ((1. - x) * (1. - y))
        return lambda x, y: x + y - x * y
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@MODELS.register_module()
class SEWBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = 'add',
                 neuron_cfg=None):
        super(SEWBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = build_node(neuron_cfg)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        from amzcls.utils.etc import get_entropy
        sorted_counter, r = get_entropy(out)
        print(r)
        print(sorted_counter)
        out = self.cnf(identity, out)
        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


@MODELS.register_module()
class SEWBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = 'add',
                 neuron_cfg=None):
        super(SEWBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = build_node(neuron_cfg)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = build_node(neuron_cfg)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)
        out = self.cnf(identity, out)
        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


default_neuron = dict(type='IFNode')
default_width = [64, 128, 256, 512]
default_stride = [1, 2, 2, 2]


@MODELS.register_module()
class SEWResNetCifar(nn.Module):
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
        self.conv1 = layer.Conv2d(in_channels, self.inplanes, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = build_node(neuron_cfg)
        self.layer1 = self._make_layer(
            block, width[0], layers[0], stride[0], cnf_list=cnf_list, neuron_cfg=neuron_cfg)
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x,

    def forward(self, x):
        return self._forward_impl(x)


def _resnet_cifar(arch, pretrained, progress, **kwargs):
    model = SEWResNetCifar(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict.pop('conv1.weight')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


@BACKBONES.register_module()
def sew_resnet18_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet18', pretrained, progress, layers=[2, 2, 2, 2], **kwargs)


@BACKBONES.register_module()
def sew_resnet34_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet34', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnet50_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet50', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnet101_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet101', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnet152_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet152', pretrained, progress, layers=[3, 8, 36, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnext50_cifar_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet_cifar('resnext50_32x4d', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnext101_cifar_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet_cifar('resnext101_32x8d', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def sew_wide_resnet50_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet50_2', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def sew_wide_resnet101_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet101_2', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@MODELS.register_module()
class SEWResNet(BaseBackbone):
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
        self.conv1 = layer.Conv2d(in_channels, self.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = build_node(neuron_cfg)
        self.maxpool = layer.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, width[0], layers[0], stride[0], cnf_list=cnf_list, neuron_cfg=neuron_cfg)
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
        # functional.set_backend(self, backend='torch', instance=NODES.get(neuron_cfg['type']))
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
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        exit(0)
        return x,

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, pretrained, progress, **kwargs):
    model = SEWResNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
        print(f'[INFO] Load pretrain weight from {model_urls[arch]}')

    return model


@BACKBONES.register_module()
def sew_resnet18(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet18', pretrained, progress, layers=[2, 2, 2, 2], **kwargs)


@BACKBONES.register_module()
def sew_resnet34(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet34', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnet50(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet50', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnet101(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet101', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnet152(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet152', pretrained, progress, layers=[3, 8, 36, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnext50_32x4d(pretrained=True, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def sew_resnext101_32x8d(pretrained=True, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def sew_wide_resnet50_2(pretrained=True, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def sew_wide_resnet101_2(pretrained=True, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)
