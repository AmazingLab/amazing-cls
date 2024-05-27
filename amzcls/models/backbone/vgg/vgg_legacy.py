import torch
from torch import nn

from ...builder import MODELS


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        factor = ctx.alpha - ctx.saved_tensors[0].abs()
        grad *= (1 / ctx.alpha) ** 2 * factor.clamp(min=0)
        return grad, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, alpha=1.0):
        super(LIFSpike, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.alpha = alpha

    def forward(self, x):
        mem_v = []
        mem = 0
        for t in range(x.shape[0]):
            mem = self.tau * mem + x[t, ...]
            spike = self.heaviside(mem - self.v_th, self.alpha)
            mem = mem * (1 - spike)
            mem_v.append(spike)

        return torch.stack(mem_v)


class StateLIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, alpha=1.0):
        super(StateLIFSpike, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.alpha = alpha
        self.mem = None
        self.have_init = False

    def init_membrane_state(self, x):
        if not self.have_init:
            # shape[T, B, C, H, W]->init_shape[1, C, H, W]
            init_shape = (1, *x.shape[2:])
            self.mem = nn.Parameter(nn.init.uniform_(
                torch.empty(init_shape, device=x.device), a=-0.2, b=0.2))
            self.have_init = True
            print('==================init==================')
        return self.mem.to(x)
        # return self.mem.sigmoid().to(x)

    def forward(self, x):
        mem = self.init_membrane_state(x)
        mem_v = []
        for t in range(x.shape[0]):
            mem = self.tau * mem + x[t, ...]
            spike = self.heaviside(mem - self.v_th, self.alpha)
            mem = mem * (1 - spike)
            mem_v.append(spike)

        return torch.stack(mem_v)


class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        time, batch = x_seq.shape[:2]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())  # [T*B, ...]
        return y_seq.view(time, batch, *y_seq.shape[1:])  # [T, B, ...]


class TEBN(nn.Module):
    def __init__(self, out_plane, time=10):
        super(TEBN, self).__init__()
        self.te = nn.Parameter(torch.ones(time, 1, 1, 1, 1))
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))

    def forward(self, x):
        return self.bn(x) * self.te


class TEBNLayer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, time=10):
        super(TEBNLayer, self).__init__()
        self.conv = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane, time)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class Layer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.layer = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


@MODELS.register_module()
class VGG11R48x48Legacy(nn.Module):
    def __init__(self, tau=0.25, time=10, num_classes=10):
        super(VGG11R48x48Legacy, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1, time),
            LIFSpike(tau=self.tau),
            TEBNLayer(64, 128, 3, 1, 1, time),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1, time),
            LIFSpike(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1, time),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(256, 512, 3, 1, 1, time),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1, time),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(512, 512, 3, 1, 1, time),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1, time),
            LIFSpike(tau=self.tau),
            pool,
        )
        w = int(48 / 2 / 2 / 2 / 2)
        self.classifier = SeqToANNContainer(
            nn.Dropout(0.25),
            nn.Linear(512 * w * w, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x,


@MODELS.register_module()
class StateVGG11R48x48Legacy(nn.Module):
    def __init__(self, tau=0.25, time=10, num_classes=10):
        super(StateVGG11R48x48Legacy, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1, time),
            StateLIFSpike(tau=self.tau),
            TEBNLayer(64, 128, 3, 1, 1, time),
            StateLIFSpike(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1, time),
            StateLIFSpike(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1, time),
            StateLIFSpike(tau=self.tau),
            pool,
            TEBNLayer(256, 512, 3, 1, 1, time),
            StateLIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1, time),
            StateLIFSpike(tau=self.tau),
            pool,
            TEBNLayer(512, 512, 3, 1, 1, time),
            StateLIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1, time),
            StateLIFSpike(tau=self.tau),
            pool,
        )
        w = int(48 / 2 / 2 / 2 / 2)
        self.classifier = SeqToANNContainer(
            nn.Dropout(0.25),
            nn.Linear(512 * w * w, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # init state
        with torch.no_grad():
            self.cuda()
            self.forward(torch.randn(1, 1, 2, 48, 48).cuda())

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x,
