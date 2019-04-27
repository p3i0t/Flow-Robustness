import torch
import torch.nn as nn
import math


class DiagonalGaussian(object):
    @staticmethod
    def sample(mean, log_std, temperature=1.):
        std = torch.exp(log_std)

        # sampling with temperature
        eps = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(std)*temperature)
        return mean + eps * std

    @staticmethod
    def logp(mean, log_std, x):
        """

        :param mean:
        :param log_std:
        :param x:
        :return:
        """
        assert mean.size() == log_std.size() == x.size(), \
            'mean {}, log_std {}, x {} should have the same size.'.format(mean.size, log_std.size(), x.size())

        std = torch.exp(log_std)
        log_prob = -0.5 * (math.log(2*math.pi) + 2*log_std + (x-mean).pow(2)/std.pow(2))
        # flatten and sum over the dims except for batch size.
        return log_prob.view(x.size(0), -1).sum(dim=1)


class GaussianPrior(object):
    """
    Diagonal Gaussian.
    """
    def __init__(self, mean, log_std):
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(self.log_std)

    def logp(self, x):
        assert self.mean.size() == self.log_std.size() == x.size(), \
            'mean {}, log_std {}, x {} should have the same size.'.format(self.mean.size, self.log_std.size(), x.size())

        log_prob = -0.5 * (math.log(2*math.pi) + 2*self.log_std + (x-self.mean).pow(2)/self.std.pow(2))
        # flatten and sum over the dims except for batch size.
        return log_prob.view(x.size(0), -1).sum(dim=1)

    def sample(self, temperature=1.):
        """
        Sample eps with temperature.
        :param temperature: temperature for sampling.
        :return:
        """
        return torch.normal(mean=self.mean, std=self.std * temperature)

    def get_eps(self, z):
        return (z - self.mean)/self.std

    def scale_eps(self, eps):
        return self.mean + eps * self.std


class LinearZeros(nn.Module):
    def __init__(self, in_size, out_size, log_scale_factor=3):
        super().__init__()
        self.log_scale_factor = log_scale_factor
        self.l = nn.Linear(in_size, out_size)

        self.l.weight.data.zero_()
        self.l.bias.data.zero_()

        self.out_size = out_size
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, out_size, 1, 1)))

    def forward(self, x):
        out = self.l(x).view(x.size(0), self.out_size, 1, 1)
        out *= torch.exp(self.log_scale * self.log_scale_factor)
        return out


get_mask = lambda h, w: torch.Tensor([[0 if (i+j) % 2 == 0 else 1 for i in range(h)] for j in range(w)]).view(1, 1, h, w)


def split_spatial(x, scheme=0):
    """
    Split a 4D Tensor across the spatial space.
    :param x: 4D tensor in BCHW format to be split.
    :return:
    """
    b, c, h, w = x.size()
    if scheme == 0:
        return x[:, :, :h // 2, :], x[:, :, h // 2:, :]
    elif scheme == 1:
        return x[:, :, h // 2:, :], x[:, :, :h // 2, :]
    elif scheme == 2:
        return x[:, :, :, :w // 2], x[:, :, :, w // 2:]
    elif scheme == 3:
        return x[:, :, :, w // 2:], x[:, :, :, :w // 2]
    # x = x.view(b, c, h*w)
    # even_idx = [i for i in range(h*w) if i % 2 == 0]
    # odd_idx = [i for i in range(h*w) if i % 2 == 1]
    # assert len(even_idx) == len(odd_idx)
    # xa, xb = x[:, :, even_idx].view(b, c, h, w//2), x[:, :, odd_idx].view(b, c, h, w//2)
    # return xa.contiguous(), xb.contiguous()


def merge_spatial(xa, xb, scheme=0):
    # assert xa.size() == xb.size()
    # b, c, h, w = xa.size()
    # xa, xb = xa.view(b, c, h*w), xb.view(b, c, h*w)
    # xa_list = xa.chunk(chunks=h*w, dim=2) # even list
    # xb_list = xb.chunk(chunks=h*w, dim=2) # odd list
    # assert len(xa_list) == len(xb_list) == h*w
    # x_list = [None] * (h*w*2)
    # x_list[::2] = xa_list
    # x_list[1::2] = xb_list
    #
    # x = torch.cat(x_list, dim=2).view(b, c, h, h)
    if scheme == 0:
        return torch.cat([xa, xb], dim=2)
    elif scheme == 1:
        return torch.cat([xb, xa], dim=2)
    elif scheme == 2:
        return torch.cat([xa, xb], dim=3)
    elif scheme == 3:
        return torch.cat([xb, xa], dim=3)

