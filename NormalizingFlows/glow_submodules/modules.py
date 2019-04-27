import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from NormalizingFlows.glow_submodules.blocks import GaussianPrior
from NormalizingFlows.glow_submodules.torchops import mean


def squeeze2d(x, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return x
    B, C, H, W = x.size()
    assert H % factor == 0 and W % factor == 0, "{} can not be factored by {}.".format((H, W), factor)
    x = x.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(x, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor_sq = factor ** 2
    if factor == 1:
        return x
    B, C, H, W = x.size()
    assert C % factor_sq == 0, "{} can not be factored by factor_sq {}".format(C, factor_sq)
    x = x.view(B, C // factor_sq, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // factor_sq, H * factor, W * factor)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x, logdet):
        return squeeze2d(x, self.factor), logdet

    def reverse(self, y):
        return unsqueeze2d(y, self.factor)


class UnsqueezeLayer(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x, logdet):
        return unsqueeze2d(x, self.factor), logdet

    def reverse(self, y):
        return squeeze2d(y, self.factor)


class ActNorm(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.n_channel = n_channel
        self.log_scale_factor = 3.
        # scale and bias as trainable parameters for data dependent initialization
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, n_channel, 1, 1)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, n_channel, 1, 1)))
        self.initialized = False

    def _init_parameters(self, x):
        if not self.training:
            return
        with torch.no_grad():
            bias = mean(x, dim=[0, 2, 3], keepdim=True)
            var = mean((x - bias)**2, dim=[0, 2, 3], keepdim=True)
            log_scale = torch.log(1./torch.sqrt(var + 1e-6))

            self.log_scale.data.copy_(log_scale.data / self.log_scale_factor)
            self.bias.data.copy_(-bias.data)
            self.initialized = True

    def forward(self, x, logdet=None):
        """

        :param x: 4D Tensor in format of (B, C, H, W)
        :param logdet: Accumulated log det from previous layers.
        :return:
        """
        if not self.initialized:
            self._init_parameters(x)

        assert x.size(1) == self.n_channel, \
            'Input n_channel {} != defined n_channel {}'.format(x.size(1), self.n_channel)
        y = (x + self.bias) * torch.exp(self.log_scale * self.log_scale_factor)  # broadcast automatically.

        if logdet is not None:
            h, w = x.size()[2:]
            dlogdet = self.logdet(h, w)
            logdet += dlogdet
            return y, logdet
        else:
            return y

    def reverse(self, y):
        x = y / torch.exp(self.log_scale * self.log_scale_factor) - self.bias
        return x

    def logdet(self, h, w):
        return h * w * (self.log_scale * self.log_scale_factor).sum()


class ConActNorm(nn.Module):
    def __init__(self, n_channel, n_classes):
        super().__init__()
        self.n_channel = n_channel
        self.log_scale_factor = 3.
        # scale and bias as trainable parameters for data dependent initialization
        self.log_scale_embed = nn.Embedding(n_classes, n_channel)
        self.bias_embed = nn.Embedding(n_classes, n_channel)
        self.initialized = False

    def _init_parameters(self, x):
        if not self.training:
            return
        with torch.no_grad():
            bias = mean(x, dim=[0, 2, 3], keepdim=True)
            var = mean((x - bias)**2, dim=[0, 2, 3], keepdim=True)
            log_scale = torch.log(1./torch.sqrt(var + 1e-6))

            bias = bias.view(1, self.n_channel)
            log_scale = log_scale.view(1, self.n_channel)

            self.log_scale_embed.weight.data.copy_(log_scale.data / self.log_scale_factor)
            self.bias_embed.weight.data.copy_(-bias.data)
            self.initialized = True

    def forward(self, x, label, logdet=None):
        """

        :param x: 4D Tensor in format of (B, C, H, W)
        :param logdet: Accumulated log det from previous layers.
        :return:
        """
        if not self.initialized:
            self._init_parameters(x)

        assert x.size(1) == self.n_channel, \
            'Input n_channel {} != defined n_channel {}'.format(x.size(1), self.n_channel)
        log_scale = self.log_scale_embed(label).view(x.size(0), self.n_channel, 1, 1)
        bias = self.bias_embed(label).view(x.size(0), self.n_channel, 1, 1)

        y = (x + bias) * torch.exp(log_scale * self.log_scale_factor)  # broadcast automatically.

        if logdet is not None:
            h, w = x.size()[2:]
            dlogdet = self.logdet(h, w, log_scale)
            logdet += dlogdet
            return y, logdet
        else:
            return y

    def reverse(self, y, label):
        log_scale = self.log_scale_embed(label).view(y.size(0), self.n_channel, 1, 1)
        bias = self.bias_embed(label).view(y.size(0), self.n_channel, 1, 1)
        x = y / torch.exp(log_scale * self.log_scale_factor) - bias
        return x

    def logdet(self, h, w, log_scale):
        return h * w * (log_scale * self.log_scale_factor).sum()


class InvertibleConv2d1x1(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        out_channels = n_channels
        in_channels = n_channels
        self.n_channels = n_channels

        weight_size = [out_channels, in_channels]
        self.register_parameter('weight', nn.Parameter(torch.zeros(*weight_size)))

        self._init_parameters()

    def _init_parameters(self):
        # initialize with a orthogonal matrix
        self.weight.data = torch.Tensor(np.linalg.qr(np.random.randn(self.n_channels, self.n_channels))[0])

    def forward(self, x, logdet):
        weight = self.weight.view(self.n_channels, self.n_channels, 1, 1)  # 2D -> 4D
        y = F.conv2d(x, weight)

        h, w = x.size()[2:]
        logdet += self.logdet(h, w, self.weight)
        return y, logdet

    def reverse(self, y):
        weight_inverse = torch.inverse(self.weight.double()).float().view(self.n_channels, self.n_channels, 1, 1)  # 2D -> 4D
        x = F.conv2d(y, weight_inverse)
        return x

    def logdet(self, h, w, weight):
        return h * w * torch.log(weight.det().abs())


class InvertibleLeakyReLU(nn.Module):
    def __init__(self, negtive_slope=0.1):
        super().__init__()
        assert 0.0 < negtive_slope < 1.0
        self.slope = negtive_slope
        self.inv_slope = 1.0 / negtive_slope

    def forward(self, x, logdet):
        return F.leaky_relu(x, self.slope), logdet + self.logdet(x)

    def reverse(self, x):
        return F.leaky_relu(x, self.inv_slope)

    def logdet(self, x):
        x = x.view(x.size(0), -1)
        mask = x > 0
        x.data.masked_fill_(mask, 1.)
        x.data.masked_fill_(1 - mask, self.slope)
        return torch.log(x).sum(dim=1)


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.padding = padding
        self.register_parameter('weight', nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size)))

        self.actnorm = ActNorm(out_channel)

        nn.init.normal_(self.weight.data, mean=0., std=0.05)

    def forward(self, x):
        if self.padding == 1:
            x = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
        out = F.conv2d(x, self.weight, padding=0)
        out = self.actnorm(out)
        return out


class ConConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, n_classes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.padding = padding
        self.register_parameter('weight', nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size)))

        self.actnorm = ConActNorm(out_channel, n_classes)

        nn.init.normal_(self.weight.data, mean=0., std=0.05)

    def forward(self, x, label):
        if self.padding == 1:
            x = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
        out = F.conv2d(x, self.weight, padding=0)
        out = self.actnorm(out, label)
        return out


class Conv2dZeros(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, log_scale_factor=1):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.log_scale_factor = log_scale_factor
        self.register_parameter('weight', nn.Parameter(torch.zeros(out_channel, in_channel, kernel_size, kernel_size)))

        self.register_parameter('bias', nn.Parameter(torch.zeros(1, out_channel, 1, 1)))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, out_channel, 1, 1)))

    def forward(self, x):
        if self.padding == 1:
            x = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
        out = F.conv2d(x, self.weight, padding=0)
        out += self.bias
        out *= torch.exp(self.log_scale * self.log_scale_factor)
        return out


class ConConv2dZeros(nn.Module):
    def __init__(self, in_channel, out_channel, n_classes, kernel_size=3, stride=1, padding=1, log_scale_factor=1):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.log_scale_factor = log_scale_factor
        self.out_channel = out_channel
        self.register_parameter('weight', nn.Parameter(torch.zeros(out_channel, in_channel, kernel_size, kernel_size)))

        # self.register_parameter('bias', nn.Parameter(torch.zeros(1, out_channel, 1, 1)))
        # self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, out_channel, 1, 1)))
        self.log_scale_embed = nn.Embedding(n_classes, out_channel)
        self.bias_embed = nn.Embedding(n_classes, out_channel)

        self.log_scale_embed.weight.data.zero_()
        self.bias_embed.weight.data.zero_()

    def forward(self, x, label):
        if self.padding == 1:
            x = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
        out = F.conv2d(x, self.weight, padding=0)

        log_scale = self.log_scale_embed(label).view(x.size(0), self.out_channel, 1, 1)
        bias = self.bias_embed(label).view(x.size(0), self.out_channel, 1, 1)

        out += bias
        out *= torch.exp(log_scale * self.log_scale_factor)
        return out


class NN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=512):
        """
        Implement the NN - nonlinear mapping, which includes 3 convolutional layers, the first and last are 3x3 and the
        middle is 1x1, with ReLU activations.
        :param in_channels:
        :param hidden_channels:
        """
        super().__init__()
        self.conv1 = Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = Conv2dZeros(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out


class ConNN(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, hidden_channels=512):
        """
        Implement the NN - nonlinear mapping, which includes 3 convolutional layers, the first and last are 3x3 and the
        middle is 1x1, with ReLU activations.
        :param in_channels:
        :param hidden_channels:
        """
        super().__init__()
        self.conv1 = ConConv2d(in_channels, hidden_channels, n_classes, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConConv2d(hidden_channels, hidden_channels, n_classes, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConConv2dZeros(hidden_channels, out_channels, n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, label):
        out = F.relu(self.conv1(x, label))
        out = F.relu(self.conv2(out, label))
        out = self.conv3(out, label)
        return out


class StepFlow(nn.Module):
    def __init__(self, in_channels, hidden_channels, permutation='conv1x1', coupling='additive'):
        super().__init__()
        self.available_permutations = ['conv1x1', 'shuffle', 'reverse']
        assert permutation in self.available_permutations, \
            'permutation {} is not supported, please choose from {}'.format(permutation, self.available_permutations)
        self.available_couplings = ['additive', 'affine']
        assert coupling in self.available_couplings, \
            'coupling {} is not supported, please choose from {}'.format(coupling, self.available_couplings)

        self.in_channels = in_channels
        self.coupling = coupling

        self.actnorm = ActNorm(in_channels)
        if permutation == 'conv1x1':
            self.permute = InvertibleConv2d1x1(in_channels)

        if coupling == 'additive':
            self.nn = NN(in_channels//2, in_channels//2, hidden_channels)
        else:
            self.nn = NN(in_channels//2, in_channels, hidden_channels)

    def forward(self, x, logdet):
        assert x.size(1) % 2 == 0
        out, logdet = self.actnorm(x, logdet)
        out, logdet = self.permute(out, logdet)

        outa, outb = out.split(self.in_channels//2, dim=1)  #BCHW format
        if self.coupling == 'additive':
            outa += self.nn(outb)  # logdet is 0
        elif self.coupling == 'affine':
            h = self.nn(outb)
            log_scale, shift = h.split(self.in_channels//2, dim=1)
            scale = torch.sigmoid(log_scale + 2.)
            outa += shift
            outa *= scale
            logdet += torch.log(scale.view(x.size(0), -1)).sum(dim=1)  # flatten and sum, keep the batch_size still.
        out = torch.cat([outa, outb], dim=1)
        return out, logdet

    def reverse(self, y):
        assert y.size(1) % 2 == 0
        x = y

        outa, outb = x.split(self.in_channels//2, dim=1)  #BCHW format
        if self.coupling == 'additive':
            outa -= self.nn(outb)  # logdet is 0
        elif self.coupling == 'affine':
            h = self.nn(outb)
            log_scale, shift = h.split(self.in_channels//2, dim=1)
            scale = torch.sigmoid(log_scale + 2.)
            outa /= scale
            outa -= shift
        x = torch.cat([outa, outb], dim=1)

        x = self.permute.reverse(x)
        x = self.actnorm.reverse(x)
        return x


class ConStepFlow(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, permutation='conv1x1', coupling='additive'):
        super().__init__()
        self.available_permutations = ['conv1x1', 'shuffle', 'reverse']
        assert permutation in self.available_permutations, \
            'permutation {} is not supported, please choose from {}'.format(permutation, self.available_permutations)
        self.available_couplings = ['additive', 'affine']
        assert coupling in self.available_couplings, \
            'coupling {} is not supported, please choose from {}'.format(coupling, self.available_couplings)

        self.in_channels = in_channels
        self.coupling = coupling

        self.actnorm = ConActNorm(in_channels, n_classes)
        if permutation == 'conv1x1':
            self.permute = InvertibleConv2d1x1(in_channels)

        if coupling == 'additive':
            self.nn = ConNN(in_channels//2, in_channels//2, n_classes, hidden_channels)
        else:
            self.nn = ConNN(in_channels//2, in_channels, n_classes, hidden_channels)

    def forward(self, x, label, logdet):
        assert x.size(1) % 2 == 0
        out, logdet = self.actnorm(x, label, logdet)
        out, logdet = self.permute(out, logdet)

        outa, outb = out.split(self.in_channels//2, dim=1)  #BCHW format
        if self.coupling == 'additive':
            outa += self.nn(outb, label)  # logdet is 0
        elif self.coupling == 'affine':
            h = self.nn(outb, label)
            log_scale, shift = h.split(self.in_channels//2, dim=1)
            scale = torch.sigmoid(log_scale + 2.)
            outa += shift
            outa *= scale
            logdet += torch.log(scale.view(x.size(0), -1)).sum(dim=1)  # flatten and sum, keep the batch_size still.
        out = torch.cat([outa, outb], dim=1)
        return out, logdet

    def reverse(self, y, label):
        assert y.size(1) % 2 == 0
        x = y

        outa, outb = x.split(self.in_channels//2, dim=1)  #BCHW format
        if self.coupling == 'additive':
            outa -= self.nn(outb, label)  # logdet is 0
        elif self.coupling == 'affine':
            h = self.nn(outb, label)
            log_scale, shift = h.split(self.in_channels//2, dim=1)
            scale = torch.sigmoid(log_scale + 2.)
            outa /= scale
            outa -= shift
        x = torch.cat([outa, outb], dim=1)

        x = self.permute.reverse(x)
        x = self.actnorm.reverse(x, label)
        return x


class Split2d(nn.Module):
    def __init__(self, in_channels):
        """
        Split the latent z to z1 and z2, factor out z1.
        :param in_channels: n_channels of the input tensor.
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(Conv2d(in_channels//2, in_channels),
                                  nn.ReLU(),
                                  Conv2dZeros(in_channels, in_channels))

        self.prior = None

    def forward(self, z, logdet):
        n_channel = self.in_channels // 2
        z1, z2 = z[:, :n_channel], z[:, n_channel:]
        out = self.conv(z1)
        mean, log_std = out[:, :n_channel], out[:, n_channel:]

        self.prior = GaussianPrior(mean, log_std)
        logdet += self.prior.logp(z2)
        eps = self.prior.get_eps(z2)
        return z1, logdet, eps

    def reverse(self, z, eps):
        assert self.prior, 'prior is None, please forward before reverse.'
        z1 = z
        z2 = self.prior.scale_eps(eps)
        z = torch.cat([z1, z2], dim=1)
        return z

    def sample(self, z, temperature):
        z1 = z
        n_channel = self.in_channels // 2
        out = self.conv(z1)
        mean, log_std = out[:, :n_channel], out[:, n_channel:]
        prior = GaussianPrior(mean, log_std)
        z2 = prior.sample(temperature=temperature)
        z = torch.cat([z1, z2], dim=1)
        return z


class FinalPrior(nn.Module):
    def __init__(self, n_channels, learn_top=True):
        super().__init__()
        self.n_channels = n_channels
        self.learn_top = learn_top

        if learn_top:
            self.conv = Conv2dZeros(2*n_channels, 2*n_channels)

        self.prior = None
        self.h = None

    def forward(self, z, logdet):
        """

        :param z: the last latent part factored out
        :param y:
        :return:
        """
        B, C, H, W = z.size()
        self.h = z.new_zeros(B, 2 * C, H, W)
        if self.learn_top:
            self.h = self.conv(self.h)

        mean, log_std = self.h[:, :self.n_channels], self.h[:, self.n_channels:]
        self.prior = GaussianPrior(mean, log_std)

        logdet += self.prior.logp(z)
        eps = self.prior.get_eps(z)
        return z, logdet, eps

    def reverse(self, eps):
        assert self.prior, 'prior is None, please forward before reverse.'
        z = self.prior.scale_eps(eps)
        return z

    def sample(self, temperature=1.):
        assert self.prior, 'prior is None, please forward before reverse.'
        return self.prior.sample(temperature=temperature)


class ConFinalPrior(nn.Module):
    def __init__(self, n_channels, n_classes, learn_top=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learn_top = learn_top

        if learn_top:
            self.conv = Conv2dZeros(2*n_channels, 2*n_channels)

        self.label_embed = nn.Embedding(n_classes, 2*n_channels)

        self.prior = None
        self.h = None

    def forward(self, z, label, logdet):
        """

        :param z: the last latent part factored out
        :param y:
        :return:
        """
        B, C, H, W = z.size()
        self.h = z.new_zeros(B, 2 * C, H, W)
        if self.learn_top:
            self.h = self.conv(self.h)

        label_out = self.label_embed(label).view(z.size(0), 2*self.n_channels, 1, 1)
        h = self.h + label_out

        mean, log_std = h[:, :self.n_channels], h[:, self.n_channels:]
        self.prior = GaussianPrior(mean, log_std)

        logdet += self.prior.logp(z)
        eps = self.prior.get_eps(z)
        return z, logdet, eps

    def reverse(self, eps, label):
        _, C, H, W = self.h.size()
        h = torch.zeros(label.size(0), C, H, W).to(label.device)
        if self.learn_top:
            h = self.conv(h)

        label_out = self.label_embed(label).view(label.size(0), 2 * self.n_channels, 1, 1)
        h = h + label_out

        mean, log_std = h[:, :self.n_channels], h[:, self.n_channels:]
        prior = GaussianPrior(mean, log_std)
        z = prior.scale_eps(eps)
        return z

    def sample(self, label, temperature=1.):
        _, C, H, W = self.h.size()
        h = torch.zeros(label.size(0), C, H, W).to(label.device)
        if self.learn_top:
            h = self.conv(h)

        label_out = self.label_embed(label).view(label.size(0), 2 * self.n_channels, 1, 1)
        h = h + label_out

        mean, log_std = h[:, :self.n_channels], h[:, self.n_channels:]
        prior = GaussianPrior(mean, log_std)
        return prior.sample(temperature=temperature)


if __name__ == '__main__':
    # x = torch.randn(2, 3, 4, 2)
    x = torch.Tensor(range(16**2)).view(1, 1, 16, 16)
    #m = NN(1, 5)
    #m = Conv2d(1, 5, kernel_size=3, padding=1)
    #print(m(x).size())
    # print(x)
    # xa, xb = split_spatial(x)
    # print(xa)
    # print(xb)
    # x = merge_spatial(xa, xb)
    # print(x)
    # print(x.size())

    # m = ActNorm(3)
    #
    # m = InvertibleConv2d1x1(3)
    # print(m(x, 0.)[0].size())
    # m = SqueezeLayer(2)
    # logdet = .0
    # out, logdet = m(x, logdet)
    # print(out.size())
    # out, logdet = m.reverse(out, logdet)
    # print(out.size())
