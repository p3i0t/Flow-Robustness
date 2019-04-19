import torch
import torch.nn as nn

from NormalizingFlows.modules import ConStepFlow, StepFlow, SqueezeLayer, Split2d, ConFinalPrior


class ConGlow(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.depth = hps.depth
        self.n_levels = hps.n_levels

        C, H, W = hps.in_channels, hps.image_size, hps.image_size

        self.layers = nn.ModuleList()
        for i in range(self.n_levels):  # scale levels
            self.layers.append(SqueezeLayer())
            C, H, W = C * 2 * 2, H // 2, W // 2
            for d in range(self.depth):  # depth of each scale levels.
                self.layers.append(StepFlow(C, hps.width, hps.permutation, hps.coupling))
            if i < self.n_levels - 1:
                self.layers.append(Split2d(C))
                C = C//2  # half is factored out
            else:
                self.layers.append(ConFinalPrior(C, hps.n_classes, hps.learn_top))

    def forward(self, x, logdet, label):
        z = x
        eps_list = []  # keep the gaussianized eps list
        for layer in self.layers:
            if isinstance(layer, Split2d):
                z, logdet, eps = layer(z, logdet)
                eps_list.append(eps)
            elif isinstance(layer, ConFinalPrior):
                z, logdet, eps = layer(z, label, logdet)
                eps_list.append(eps)
            # elif isinstance(layer, ConStepFlow):
            #     z, logdet = layer(z, label, logdet)
            else:
                z, logdet = layer(z, logdet)
        return z, logdet, eps_list

    def reverse(self, eps_list, label):
        with torch.no_grad():  # No update will be performed.
            for layer in reversed(self.layers):
                if isinstance(layer, ConFinalPrior):
                    eps = eps_list.pop()
                    z = layer.reverse(eps, label)
                elif isinstance(layer, Split2d):
                    eps = eps_list.pop()
                    z = layer.reverse(z, eps)
                # elif isinstance(layer, ConStepFlow):
                #     z = layer.reverse(z, label)
                else:
                    z = layer.reverse(z)
        x = z
        return x

    def sample(self, label, temperature=1.):
        with torch.no_grad():  # No update will be performed.
            for layer in reversed(self.layers):
                if isinstance(layer, ConFinalPrior):
                    z = layer.sample(label=label, temperature=temperature)
                elif isinstance(layer, Split2d):
                    z = layer.sample(z, temperature=temperature)
                # elif isinstance(layer, StepFlow):
                #     z = layer.reverse(z, label)
                else:
                    z = layer.reverse(z)
        x = z
        return x


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    import sys
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", type=str, default='mnist',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str,
                        default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=128,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=2,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=2,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=2,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learn_top", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")

    hps = parser.parse_args()  # So error if typo
    # main(hps)
    image_size = 8
    depth = 1
    n_levels = 1
    hidden_channels = 8
    logdet = torch.zeros(5)

    x = torch.randn(5, 3, image_size, image_size)
    hps.in_channels = 3
    hps.hidden_channels = 3
    hps.permutation = 'conv1x1'
    hps.coupling = 'additive'
    hps.learn_top = False
    hps.y_condition = False

    m = ConGlow(hps)
    # print('model: ', m)
    z, logdet, eps_list = m(x, logdet)
    x_reverse = m.reverse(eps_list)
    print('logdet ', logdet.size())
    print('x_reverse == x? : ', torch.allclose(x, x_reverse, atol=1e-6))


