import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.optim import Adam

import argparse
import os
import numpy as np
from NormalizingFlows import Glow, ConGlow


def preprocess(x):
    x = x * (hps.n_bins - 1)
    x = x / hps.n_bins - 0.5
    return x


def postprocess(x):
    return (x + 0.5).clamp(0., 1.)


def one_hot(y, n_classes):
    emb = torch.eye(n_classes)
    return emb[y.long()]


def get_dataset(dataset='mnist', train=True, class_id=None):
    if dataset == 'mnist':
        dataset = datasets.MNIST('data/MNIST', train=train, download=True,
                                 transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                   ]))
    elif dataset == 'fashion':
        dataset = datasets.FashionMNIST('data/FashionMNIST', train=train, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                        ]))
    else:
        print('dataset {} is not available'.format(dataset))

    if class_id != -1:
        class_id = int(class_id)
        if train:
            idx = (dataset.train_labels == class_id)
            dataset.train_labels = dataset.train_labels[idx]
            dataset.train_data = dataset.train_data[idx]
        else:
            idx = (dataset.test_labels == class_id)
            dataset.test_labels = dataset.test_labels[idx]
            dataset.test_data = dataset.test_data[idx]
    return dataset


def _weights_init(m):
    # classname = m.__class__.__name__
    # # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channel//4, out_channel//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channel, self.expansion * out_channel, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * out_channel)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channel=1):
        super(ResNet, self).__init__()
        self.in_channel = 32

        multiplier = self.in_channel

        self.conv1 = nn.Conv2d(image_channel, multiplier, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(multiplier)

        # 4 stages resnet
        self.layer1 = self._make_layer(block, multiplier, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, multiplier * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, multiplier * 4, num_blocks[2], stride=2)
        self.linear = nn.Sequential(nn.Linear(multiplier * 4, multiplier * 4),
                                    nn.BatchNorm1d(multiplier * 4),
                                    nn.ReLU(),
                                    nn.Linear(multiplier * 4, num_classes))

        self.apply(_weights_init)

    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def build_resnet_32x32(n=8, fc_size=10, image_channel=3):
    assert (n - 2) % 6 == 0, '{} should be expressed in form of 6n+n'.format(n)
    block_depth = int((n - 2) / 6)
    return ResNet(BasicBlock, [block_depth]*3, num_classes=fc_size, image_channel=image_channel)


def train_classifier(classifier, hps):
    dataset = get_dataset(dataset=hps.problem, train=True, class_id=hps.class_id)
    train_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_train, shuffle=True)

    dataset = get_dataset(dataset=hps.problem, train=False, class_id=hps.class_id)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=True)

    optimizer = Adam(classifier.parameters(), lr=0.001)

    classifier.train()
    for epoch in range(1, hps.epochs+1):
        print('Epoch {}'.format(epoch))
        for batch_id, (x, y) in enumerate(train_loader):
            x = preprocess(x).to(hps.device)
            y = y.to(hps.device)

            logits = classifier(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = preprocess(x).to(hps.device)
        y = y.to(hps.device)

        logits = classifier(x)
        acc = (torch.argmax(logits, dim=-1) == y).float().mean().item()
        acc_list.append(acc)
    print('Test acc: {:.4f}'.format(np.mean(acc_list)))

    torch.save(classifier.state_dict(), os.path.join(hps.log_dir, 'classifier_{}.pth'.format(hps.problem)))


def eval_classifier(classifier, hps):
    dataset = get_dataset(dataset=hps.problem, train=False, class_id=hps.class_id)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=True)

    path = os.path.join(hps.log_dir, 'classifier_{}.pth'.format(hps.problem))
    classifier.load_state_dict(torch.load(path))

    def left_shift(x, n_pixel=1):
        return torch.cat([x[:, :, :, n_pixel:], x[:, :, :, :n_pixel]], dim=-1)

    classifier.eval()
    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = preprocess(x).to(hps.device)
        x = left_shift(x)
        y = y.to(hps.device)

        logits = classifier(x)
        acc = (torch.argmax(logits, dim=-1) == y).float().mean().item()
        acc_list.append(acc)
    print('1-pixel left shift, Test acc: {:.4f}'.format(np.mean(acc_list)))

    acc_list = []
    eps = 0.001
    for batch_id, (x, y) in enumerate(test_loader):
        x = preprocess(x).to(hps.device)
        x = x + eps * torch.randn(x.size()).to(hps.device)
        y = y.to(hps.device)

        logits = classifier(x)
        acc = (torch.argmax(logits, dim=-1) == y).float().mean().item()
        acc_list.append(acc)
    print('Gaussian Noises, Test acc: {:.4f}'.format(np.mean(acc_list)))

    glow = ConGlow(hps).to(hps.device)

    suffix = '' if hps.class_id == -1 else '_{}'.format(hps.class_id)
    checkpoint = torch.load(os.path.join(hps.log_dir, '{}_glow_{}{}.pth'.format(hps.coupling, hps.problem, suffix))
                            , map_location=lambda storage, loc: storage)
    glow.load_state_dict(checkpoint['model_state'])
    glow.eval()

    def f(x, y):
        loglikelihood = torch.zeros(x.size(0)).to(hps.device)

        n_pixels = np.prod(x.size()[1:])
        loglikelihood += -np.log(hps.n_bins) * n_pixels
        z, loglikelihood, eps_list = glow(x, loglikelihood, y)

        bits_x = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
        return bits_x, eps_list

    def zero_epses(eps_list, n_zeros=1):
        "Zero the preceding n_zeros eps factors. "
        assert n_zeros <= len(eps_list)
        for idx in range(n_zeros):
            eps_list[idx].zero_()
        return eps_list

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = preprocess(x).to(hps.device)
        bits, eps_list = f(x, y)
        zeroed_eps_list = zero_epses(eps_list, n_zeros=1)
        x_reverse = glow.reverse(zeroed_eps_list, y)

        y = y.to(hps.device)

        logits = classifier(x_reverse)
        acc = (torch.argmax(logits, dim=-1) == y).float().mean().item()
        acc_list.append(acc)
    print('Zeroing <1, Test acc: {:.4f}'.format(np.mean(acc_list)))

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = preprocess(x).to(hps.device)
        bits, eps_list = f(x, y)
        zeroed_eps_list = zero_epses(eps_list, n_zeros=2)
        x_reverse = glow.reverse(zeroed_eps_list, y)

        y = y.to(hps.device)

        logits = classifier(x_reverse)
        acc = (torch.argmax(logits, dim=-1) == y).float().mean().item()
        acc_list.append(acc)
    print('Zeroing <2, Test acc: {:.4f}'.format(np.mean(acc_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--inference", action="store_true",
                        help="Use in inference mode")
    parser.add_argument("--translation_attack", action="store_true",
                        help="perform translation attack")
    parser.add_argument("--reverse_attack", action="store_true",
                        help="perform reverse attack")
    parser.add_argument("--gradient_attack", action="store_true",
                        help="perform gradient attack")
    parser.add_argument("--sample", action="store_true",
                        help="Use in sample mode")
    parser.add_argument("--log_dir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='mnist',
                        help="Problem (mnist/fashion/cifar10/imagenet")
    parser.add_argument("--n_classes", type=int,
                        default=10, help="number of classes of dataset.")
    parser.add_argument("--infer_problem", type=str, default='mnist',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--class_id", type=int,
                        default=-1, help="single class_id for training.")
    parser.add_argument("--infer_class_id", type=int,
                        default=-1, help="single class_id for inference.")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Location of data")

    # Optimization hyperparams:
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=100, help="Minibatch size")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total number of training epochs")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--width", type=int, default=128,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=8,
                        help="Depth of network")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=5,
                        help="Number of levels")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=64,
                        help="minibatch size for sample")

    # Ablation
    parser.add_argument("--learn_top", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--permutation", type=str, default='conv1x1',
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--coupling", type=str, default='affine',
                        help="Coupling type: 0=additive, 1=affine")

    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")
    hps.n_bins = 2. ** hps.n_bits_x  # number of pixel levels

    hps.in_channels = 1 if hps.problem == 'mnist' or hps.problem == 'fashion' else 3
    hps.hidden_channels = hps.width

    m = build_resnet_32x32(image_channel=1).to(hps.device)
    train_classifier(m, hps) 
