import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _ordering_sampling(input_size, hidden_sizes, input_order='default'):
    orders = [np.arange(1, input_size+1)]
    if input_order == 'random':
        np.random.shuffle(orders[0])
    elif input_order == 'reverse':
        orders[0] = orders[0][::-1]

    for l, h_size in enumerate(hidden_sizes):
        min_prev = min(np.min(orders[-1]), input_size-1)
        order = np.random.randint(min_prev, input_size, h_size)
        orders.append(order)
    return orders


def _gen_mask(orders):
    masks = []
    for l, (d1, d2) in enumerate(zip(orders[:-1], orders[1:])):
        m = (np.expand_dims(d1, axis=1) <= d2).astype(np.float32)
        masks.append(torch.FloatTensor(m))
    m = (np.expand_dims(orders[-1], axis=1) < orders[0]).astype(np.float32)
    masks.append(torch.FloatTensor(m))
    return masks

#
# class MyWNLinear(nn.Linear):
#     def __init__(self, in_size, out_size, bias=True):
#         super(MyWNLinear, self).__init__(in_size, out_size, bias)
#         limit = math.sqrt(6/(in_size + out_size))
#         nn.init.uniform_(self.weight, -limit, limit)
#
#     def forward(self, _input):
#         return F.linear(_input, self.weight, self.bias)


class MaskedLinear(nn.Linear):
    """Masked Linear layer based on nn.Linear"""
    def __init__(self, in_size, out_size, mask, bias=True):
        super(MaskedLinear, self).__init__(in_size, out_size, bias)
        mask = nn.Parameter(mask, requires_grad=False)
        self.register_parameter('mask', mask)

    def forward(self, _input):
        """ forward method of masked linear"""
        masked_weight = self.weight * self.mask
        return F.linear(_input, masked_weight, self.bias)


# class BatchNorm(nn.Module):
#     def __init__(self, feature_size, expand_size):
#         super(BatchNorm, self).__init__()
#         self.vars_created = False
#         self.eps = 1e-6
#         self.decay = .99
#
#         self.beta = nn.Parameter(torch.zeros(1, feature_size, expand_size))
#         self.gamma = nn.Parameter(torch.ones(1, feature_size, expand_size))
#         self.train_m = None
#         self.train_v = None
#
#     def forward(self, x):
#         mu = torch.mean(x, dim=0, keepdim=True)
#         var = torch.mean((x - mu)**2, dim=0, keepdim=True) + self.eps
#         self.train_m = mu
#         self.train_v = var
#
#         out = (x-self.train_m)/torch.sqrt(self.train_v)
#         x_bn = out * torch.sqrt(self.gamma) + self.beta
#
#         return x_bn


class AutoRegressiveNN(nn.Module):
    """
    A basic MADE building block enforcing autoregressive property.
    """
    def __init__(self, in_size, hidden_sizes, out_size_multiplier=2, input_order='reverse'):
        super(AutoRegressiveNN, self).__init__()
        self.in_size = in_size
        self.out_size_multiplier = out_size_multiplier

        # Generate and repeat masks
        degrees = _ordering_sampling(in_size, hidden_sizes, input_order)
        self.input_order = degrees[0]
        masks = _gen_mask(degrees)
        sizes = [in_size] + list(hidden_sizes)
        self.module_list = nn.ModuleList([])
        for l, (i_size, o_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.module_list.append(MaskedLinear(i_size, o_size, masks[l].t_()))
            self.module_list.append(nn.BatchNorm1d(o_size))
            self.module_list.append(nn.ELU())

        final_mask = masks[-1].clone().repeat(1, out_size_multiplier)
        self.module_list.append(MaskedLinear(sizes[-1], out_size_multiplier*in_size, final_mask.t_()))

    def forward(self, _input):
        out = _input
        for l, layer in enumerate(self.module_list):
            out = layer(out)
        return out

if __name__ == '__main__':
    m = AutoRegressiveNN(3, [5, ], out_size_multiplier=2)
    x = torch.randn(7, 3)
    # out = m(x)
    # print(out.size())
