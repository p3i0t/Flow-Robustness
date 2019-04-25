#! /usr/bin/env python

import os
import time
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


def get_loader(args):
    dir = os.path.join(args.data_dir, args.problem)
    if args.problem == 'mnist':
        tr = data.DataLoader(datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor()),
                             batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        te = data.DataLoader(datasets.MNIST(dir, train=False, download=True, transform=transforms.ToTensor()),
                             batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    elif args.problem == 'fashion':
        tr = data.DataLoader(datasets.FashionMNIST(dir, train=True, download=True, transform=transforms.ToTensor()),
                             batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        te = data.DataLoader(datasets.FashionMNIST(dir, train=False, download=True, transform=transforms.ToTensor()),
                             batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    return tr, te


def train(model, args):
    model_name = 'pcnn_{}'.format(args.problem)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = get_loader(args)

    optimal_loss = 1e4
    for epoch in range(1, args.epochs+1):
        err_tr = []
        time_tr = time.time()
        model.train()
        for x, y in train_loader:
            x = x.to(args.device)
            target = (x.data[:, 0] * 255).long().to(args.device)
            loss = F.cross_entropy(model(x), target)
            err_tr.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_tr = time.time() - time_tr

        # compute error on test set
        err_te = []
        cuda.synchronize()
        time_te = time.time()
        model.eval()
        for x, y in train_loader:
            x = x.to(args.device)
            target = (x.data[:, 0] * 255).long().to(args.device)
            loss = F.cross_entropy(model(x), target)
            err_te.append(loss.item())
        cuda.synchronize()
        time_te = time.time() - time_te

        if np.mean(err_tr) < optimal_loss:
            optimal_loss = np.mean(err_tr)
            print('==> new SOTA achieved, saving model ...')
            torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pth'.format(model_name)))

        print('epoch={}; nll_tr={:.4f}; nll_te={:.4f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
                epoch, np.mean(err_tr), np.mean(err_te), time_tr, time_te))

    # # sample
    # sample.fill_(0)
    # net.train(False)
    # for i in range(28):
    #     for j in range(28):
    #         out = net(Variable(sample, volatile=True))
    #         probs = F.softmax(out[:, :, i, j]).data
    #         sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
    # utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12, padding=0)
    #


def translation_attack(model, args):
    model_name = 'pcnn_{}'.format(args.problem)
    state_dict = torch.load(os.path.join(args.save_dir, '{}.pth'.format(model_name)),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    args.batch_size = 1
    train_loader, test_loader = get_loader(args)

    def f(x):
        target = (x.data[:, 0] * 255).long().to(args.device)
        loss = F.cross_entropy(model(x), target, reduction='mean')  # keep batch dim
        print('loss: ', loss.size(), loss)
        exit(0)
        return loss

    def eval_bits(data_loader):
        for batch_id, (x, y) in enumerate(data_loader):
            x = x.to(args.device)
            bits = f(x)
    
    with torch.no_grad():
        eval_bits(test_loader)



def gradient_attack(model, args):
    model_name = 'pcnn_{}'.format(args.problem)
    state_dict = torch.load(os.path.join(save_dir, '{}.pth'.format(model_name)),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    args.batch_size = 1
    train_loader, test_loader = get_loader(args)

    def f(x):
        target = (x.data[:, 0] * 255).long().to(args.device)
        loss = F.cross_entropy(model(input), target)

        grad = torch.autograd.grad(outputs=loss,
                                   inputs=x,
                                   grad_outputs=torch.ones(loss.size()).to(args.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        return loss.item(), grad

    n_iterations = 10
    step = 1e-3

    mode = 'descent'
    n_samples = 1

    for batch_id, (x, y) in enumerate(test_loader):
        if batch_id == n_samples:
            break

        mask = (x < 0.5).to(args.device)
        x.requires_grad = True
        x = x.to(args.device)
        x_original = x

        for i in range(n_iterations):
            loss, x_grad = f(x)
            # print('grad sum: ', x_grad.sum().item())
            if i == 0:
                print('initial gradient loss: {:.4f}'.format(loss))
                utils.save_image(x,
                                 os.path.join(save_dir,
                                              'pixelcnn_gradient_{}_original{}.png'.format(args.problem, batch_id)))
            x = x + step * x_grad
            # x = x + step * mask.float() * x_grad
            # if mode == 'descent':
            #     x = x - step * x_grad
            # elif mode == 'ascent':
            #     x = x + step * x_grad

        print('gradient bpd: {:.4f}'.format(loss))
        utils.save_image(x,
                         os.path.join(save_dir,
                                      'pixelcnn_gradient_{}_adversarial{}.png'.format(mode, args.problem, batch_id)))

        x = x_original
        for i in range(n_iterations):
            loss, x_grad = f(x)
            x = x + step * mask.float() * x_grad
            # if mode == 'descent':
            #     x = x - step * x_grad
            # elif mode == 'ascent':
            #     x = x + step * x_grad

        print('gradient bpd: {:.4f}'.format(loss))
        utils.save_image(x,
                         os.path.join(save_dir,
                                      'pixelcnn_gradient_{}_adversarial{}_mask.png'.format(mode, args.problem, batch_id)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='logs',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-p', '--problem', type=str,
                        default='mnist', help='Can be either cifar|mnist')
    # parser.add_argument('-p', '--print_every', type=int, default=50,
    #                     help='how many iterations between print statements')
    parser.add_argument('-t', '--save_interval', type=int, default=1,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=2,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=100,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=1,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--gradient_attack", action="store_true",
                        help="use gradient attack")
    parser.add_argument("--translation_attack", action="store_true",
                        help="use translation attack")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total number of training epochs")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")

    fm = 64
    model = nn.Sequential(
        MaskedConv2d('A', 1, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        nn.Conv2d(fm, 256, 1))

    model = model.to(args.device)

    if args.gradient_attack:
        gradient_attack(model, args)
    elif args.translation_attack:
        translation_attack(model, args)
    else:
        train(model, args)
