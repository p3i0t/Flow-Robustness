import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim import Adam, Adamax

from NormalizingFlows import MaskedAutoregressiveFlow


def flatten(x):
    return x.view(x.size(0), -1)


def unflatten(x, size=(1, 28, 28)):
    return x.view(x.size(0), *size)


def preprocess(x):
    return flatten(x) - 0.5


def postprocess(x):
    return (unflatten(x) + 0.5).clamp(0., 1.)


def get_dataset(dataset='mnist', train=True, class_id=None):
    if dataset == 'mnist':
        dataset = datasets.MNIST('data/MNIST', train=train, download=True,
                                 transform=transforms.Compose([
                                       #transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                   ]))
    elif dataset == 'fashion':
        dataset = datasets.FashionMNIST('data/FashionMNIST', train=train, download=True,
                                        transform=transforms.Compose([
                                            #transforms.Resize((32, 32)),
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


def train(maf, optimizer, hps):
    maf.train()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    # Create log dir
    logdir = os.path.abspath(hps.log_dir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    dataset = get_dataset(dataset=hps.problem, train=True, class_id=hps.class_id)
    train_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_train, shuffle=True)

    best_bits_per_dim = np.inf
    for epoch in range(1, hps.epochs+1):
        bits_list = []

        for batch_id, (x, y) in enumerate(train_loader):
            x = preprocess(x).to(hps.device)
            x = x + torch.empty(x.size()).uniform_(0, 1/hps.n_bins).to(hps.device)  # add small uniform noise

            y = y.to(hps.device)
            loglikelihood = torch.zeros(x.size(0)).to(hps.device)

            n_pixels = np.prod(x.size()[1:])
            loglikelihood += -np.log(hps.n_bins) * n_pixels

            optimizer.zero_grad()
            log_probs, u = maf(x)

            loglikelihood += log_probs

            # Generative loss
            bits_x = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
            mean_bits_x = bits_x.mean()
            mean_bits_x.backward()
            optimizer.step()

            bits_list.append(mean_bits_x.cpu().item())

        # sampling images.
        save_image(postprocess(x), os.path.join(hps.log_dir, 'maf_epoch{}_original.png'.format(epoch)))
        x_reverse = maf.reverse(u)
        save_image(postprocess(x_reverse), os.path.join(hps.log_dir, 'maf_epoch{}_reverse.png'.format(epoch)))
        x_sample = maf.reverse(torch.randn(u.size()).to(hps.device))
        save_image(postprocess(x_sample), os.path.join(hps.log_dir, 'maf_epoch{}_sample.png'.format(epoch)))

        cur_bits_per_dim = np.mean(bits_list)
        print('Epoch {}, mean bits_per_dim: {:.4f}'.format(epoch, cur_bits_per_dim))

        if cur_bits_per_dim < best_bits_per_dim:
            best_bits_per_dim = cur_bits_per_dim
            checkpoint = {'model_state': maf.state_dict(),
                          'bits_per_dim': best_bits_per_dim,
                          'hps': hps
                          }
            suffix = '' if hps.class_id == -1 else '_{}'.format(hps.class_id)
            torch.save(checkpoint, os.path.join(hps.log_dir, 'maf_{}{}.pth'.format(hps.problem, suffix)))
            print('==> New optimal model saved !!!')


if __name__ == "__main__":
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

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
                        default=50, help="Minibatch size")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total number of training epochs")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--width", type=int, default=128,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=8,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_y", type=int, default=10,
                        help="Weight of log p(y|x) in weighted loss")
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
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")
    hps.n_bins = 2. ** hps.n_bits_x  # number of pixel levels

    hps.in_channels = 1 if hps.problem == 'mnist' or hps.problem == 'fashion' else 3
    hps.in_size = 28 * 28
    hps.hidden_sizes = [1024, 1024]
    hps.n_mades = 2

    maf = MaskedAutoregressiveFlow(hps.in_size, hps.hidden_sizes, n_mades=hps.n_mades).to(hps.device)
    optimizer = Adam(maf.parameters(), lr=hps.lr)

    # if hps.inference:
    #     inference(maf, hps)
    # elif hps.translation_attack:
    #     translation_attack(maf, hps)
    # elif hps.reverse_attack:
    #     reverse_attack(maf, hps)
    # elif hps.gradient_attack:
    #     gradient_attack(maf, hps)
    # else:
    train(maf, optimizer, hps)


