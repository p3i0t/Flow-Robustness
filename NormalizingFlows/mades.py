import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import math


from NormalizingFlows.auto_regressive import AutoRegressiveNN


class GaussianMade(nn.Module):
    def __init__(self, in_size, hidden_sizes, input_order='reverse'):
        super(GaussianMade, self).__init__()
        self._in_size = in_size
        self._hidden_sizes = hidden_sizes

        self.ar = AutoRegressiveNN(in_size, hidden_sizes, out_size_multiplier=2, input_order=input_order)

    def reverse(self, u):
        """
            Sample method of MAF, which takes in_size passes to finish
        :param n_samples: number of samples to generate
        :param u: random seed for generation, if u is None, generate a random seed.
        :return: return n_samples samples.
        """
        with torch.no_grad():
            n_samples = u.size(0)
            device = next(self.ar.parameters()).device  # infer which device this module is on now.
            x = torch.zeros([n_samples, self._in_size]).to(device)
            #u = torch.randn((n_samples, self._in_size))
            # if torch.cuda.is_available():
            #     x = x.cuda()
            #     u = u.cuda()

            for i in range(1, self._in_size+1):
                mu, log_sig_sq = torch.split(self.ar(x), split_size_or_sections=self._in_size, dim=1)
                ind = (self.ar.input_order == i).nonzero()
                sig = torch.exp(torch.clamp(0.5*log_sig_sq[:, ind], max=5.0))
                x[:, ind] = u[:, ind]*sig + mu[:, ind]
        return x

    def forward(self, x):
        out = self.ar(x)
        mu, log_sigma_sq = torch.split(out, split_size_or_sections=self._in_size, dim=1)
        u = (x - mu) * torch.exp(-0.5 * log_sigma_sq)
        # log_probs = torch.sum(-0.5 * (math.log(2 * math.pi) + log_sigma_sq + u**2), dim=1)

        log_det_du_dx = torch.sum(-0.5 * log_sigma_sq, dim=1)
        return log_det_du_dx, u

if __name__ == '__main__':
    m = GaussianMade(3, [5, ])
    x = torch.randn(7, 3)
    log_det, u = m(x)
    print(log_det.size(), u.size())
    x_reverse = m.reverse(u)
    diff = (x - x_reverse).sum()

    print(diff)
