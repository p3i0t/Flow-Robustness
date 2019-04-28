import time
import os
import argparse
import torch

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from NormalizingFlows.pixelcnn_utils import discretized_mix_logistic_loss, discretized_mix_logistic_loss_1d, \
    sample_from_discretized_mix_logistic, sample_from_discretized_mix_logistic_1d
from NormalizingFlows.pixelcnn import PixelCNN


import numpy as np


rescaling = lambda x: (x - .5) * 2.
rescaling_inv = lambda x: .5 * x + .5


def sample(model, args):
    train_loader, test_loader, loss_op, sample_op = get_dataset_ops(args)
    model.eval()
    data = torch.zeros(args.sample_batch_size, args.obs[0], args.obs[1], args.obs[2])
    data = data.to(args.device)

    for i in range(args.obs[1]):
        for j in range(args.obs[2]):
            # data_v = Variable(data, volatile=True)
            out = model(data, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data


def train(model, args):
    model_name = 'pixelcnn_{}'.format(args.problem)

    train_set, test_set, loss_op, sample_op = get_dataset_ops(args)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    print('starting training')

    deno = args.batch_size * np.prod(args.obs) * np.log(2.)

    def f(x):
        """ Calculate bpd, average over per batch samples."""
        output = model(x)
        bpd = loss_op(x, output) / deno
        return bpd

    optimal_bpd = 1e4

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x.requires_grad = True
            x = x.to(args.device)

            bpd = f(x)
            optimizer.zero_grad()
            bpd.backward()
            optimizer.step()

        # decrease learning rate
        scheduler.step()

        model.eval()
        bpd_list = []
        for batch_idx, (x, _) in enumerate(test_loader):
            x.requires_grad = True
            x = x.to(args.device)
            bpd = f(x)
            bpd_list.append(bpd.item())
        print('test bpd: {:.4f}'.format(np.mean(bpd_list)))

        if np.mean(bpd_list) < optimal_bpd:
            optimal_bpd = np.mean(bpd_list)

            check_point = {'state_dict': model.state_dict(),
                           'args': args,
                           'optimal_bpd': optimal_bpd}

            torch.save(check_point, os.path.join(args.log_dir, '{}.pth'.format(model_name)))

            print('sampling...')
            sample_t = sample(model, args)
            sample_t = rescaling_inv(sample_t)
            utils.save_image(sample_t, os.path.join(args.log_dir, '{}_{}.png'.format(model_name, epoch)),
                             nrow=5, padding=0)


def get_dataset_ops(args):
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    if args.problem == 'mnist':
        dir = os.path.join(args.data_dir, 'MNIST')
        train_set = datasets.MNIST(dir, download=True, train=True, transform=ds_transforms)
        test_set = datasets.MNIST(dir, train=False, transform=ds_transforms)

        loss_op = lambda real, fake: discretized_mix_logistic_loss_1d(real, fake)
        sample_op = lambda x: sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

    elif args.problem == 'fashion':
        dir = os.path.join(args.data_dir, 'FashionMNIST')
        train_set = datasets.FashionMNIST(dir, download=True, train=True, transform=ds_transforms)
        test_set = datasets.FashionMNIST(dir, train=False, transform=ds_transforms)

        loss_op = lambda real, fake: discretized_mix_logistic_loss_1d(real, fake)
        sample_op = lambda x: sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)
    else:
        raise Exception('{} dataset not in [mnist, fashion]'.format(args.dataset))

    return train_set, test_set, loss_op, sample_op


def translation_attack(model, args):
    model.eval()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = 'pixelcnn_{}'.format(args.problem)

    args.batch_size = 1
    train_set, test_set, loss_op, sample_op = get_dataset_ops(args)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

    save_dir = os.path.join(args.log_dir, 'translation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    check_point = torch.load(os.path.join(args.log_dir, '{}.pth'.format(model_name)),
                             map_location=lambda storage, loc: storage)
    model.load_state_dict(check_point['state_dict'])

    deno = args.batch_size * np.prod(args.obs) * np.log(2.)

    def f(x):
        output = model(x)
        loss = loss_op(x, output)
        bpd = loss / deno
        return bpd

    def left_shift(x, n_pixel=1):
        return torch.cat([x[:, :, :, n_pixel:], x[:, :, :, :n_pixel]], dim=-1)

    def up_shift(x, n_pixel=1):
        return torch.cat([x[:, :, n_pixel:, :], x[:, :, :n_pixel, :]], dim=-2)

    def left_up(x, n_pixel=1):
        x = left_shift(x, n_pixel)
        x = up_shift(x, n_pixel)
        return x

    def eval_bits(data_loader, args, left_pixel=None):
        pix = 0 if left_pixel is None else left_pixel
        keys = ['{}_class{}_leftpixel{}'.format(args.problem, i, pix) for i in range(args.n_classes)]
        bits_dict = {key: list() for key in keys}

        for batch_id, (x, y) in enumerate(data_loader):
            #x = rescaling_inv(x).to(args.device)
            x = rescaling_inv(x).to(args.device)
            y = y.to(args.device)
            if left_pixel:
                x = left_shift(x, left_pixel)
                if args.problem == 'mnist':
                    x = up_shift(x, left_pixel)
            bits_x = f(x)
            bits_dict['{}_class{}_leftpixel{}'.format(args.problem, y.item(), pix)].append(bits_x.item())
        return bits_dict

    with torch.no_grad():
        # Evaluate on test set with different pixel shifts to left
        bits_dict = {}
        bits = eval_bits(test_loader, args)
        bits_dict.update(bits)

        left_bits_1 = eval_bits(test_loader, args, left_pixel=1)
        left_bits_2 = eval_bits(test_loader, args, left_pixel=2)
        bits_dict.update(left_bits_1)
        bits_dict.update(left_bits_2)

        torch.save(bits_dict, os.path.join(save_dir, 'pcnn_{}_bits_dict.pth'.format(args.problem)))

        n_samples = 4

        for sample_id, (x, y) in enumerate(test_loader):
            if sample_id == n_samples:
                break

            x = rescaling(x).to(args.device)
            y = y.to(args.device)

            bits_x = f(x)
            utils.save_image(rescaling_inv(x), os.path.join(save_dir, 'pcnn_{}_original_{}_bpd[{:.3f}].png'.format(
                args.problem, sample_id, bits_x.cpu().item())))

            x = left_shift(x, n_pixel=1)
            bits_x = f(x)
            utils.save_image(rescaling_inv(x), os.path.join(save_dir, 'pcnn_{}_l1_{}_bpd[{:.3f}].png'.format(
                args.problem, sample_id, bits_x.cpu().item())))

            x = left_shift(x, n_pixel=1)
            bits_x = f(x)
            utils.save_image(rescaling_inv(x), os.path.join(save_dir, 'pcnn_{}_l2_{}_bpd[{:.3f}].png'.format(
                args.problem, sample_id, bits_x.cpu().item())))


def gradient_attack(model, args):
    model.eval()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = 'pixelcnn_{}'.format(args.problem)
    check_point = torch.load(os.path.join(args.log_dir, '{}.pth'.format(model_name)),
                             map_location=lambda storage, loc: storage)
    model.load_state_dict(check_point['state_dict'])

    args.batch_size = 1
    train_set, test_set, loss_op, sample_op = get_dataset_ops(args)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

    save_dir = os.path.join(args.log_dir, 'gradient')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    deno = args.batch_size * np.prod(args.obs) * np.log(2.)

    def f(x):
        output = model(x)
        bpd = loss_op(x, output) / deno
        grad = torch.autograd.grad(outputs=bpd,
                                   inputs=x,
                                   grad_outputs=torch.ones(bpd.size()).to(args.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        return bpd.item(), grad

    n_iterations = 10
    step = 1e-2

    n_samples = 4

    for batch_id, (x, y) in enumerate(test_loader):
        if batch_id == n_samples:
            break
        mask = (x < 0.4).to(args.device)

        x.requires_grad = True
        x = x.to(args.device)
        y = y.to(args.device)

        x_original = x

        for i in range(n_iterations):
            bpd, x_grad = f(x)
            if i == 0:
                print('original image bpd: {:.4f}'.format(bpd))
                utils.save_image(rescaling_inv(x),
                           os.path.join(save_dir,
                                        'pixelcnn_{}_original_{}_bpd[{:.3f}].png'.format(args.problem, batch_id, bpd)))
            x = x + step * x_grad
        print('gradient bpd: {:.4f}'.format(bpd))
        utils.save_image(rescaling_inv(x),
                   os.path.join(save_dir, 'pixelcnn_{}_gradient{}_bpd[{:.3f}].png'.format(args.problem, batch_id, bpd)))

        # diff = x - x_original
        # diff *= 1000
        # utils.save_image(diff,
        #                  os.path.join(save_dir, 'pixelcnn_{}_noise_{}.png'.
        #                               format(args.problem, batch_id)), normalize=True)

        x = x_original
        for i in range(n_iterations):
            bpd, x_grad = f(x)
            x = x + step * mask.float() * x_grad

        print('gradient bpd: {:.4f}'.format(bpd))
        utils.save_image(rescaling_inv(x),
                   os.path.join(save_dir,
                                'pixelcnn_{}_mask{}_bpd[{:.3f}].png'.format(args.problem, batch_id, bpd)))

        x = x_original
        x = x + step * torch.randn(x_grad.size()).to(args.device)   # random noise
        bpd, x_grad = f(x)
        print('gradient bpd: {:.4f}'.format(bpd))
        utils.save_image(rescaling_inv(x),
                         os.path.join(save_dir,
                                      'pixelcnn_{}_randnoise_{}_bpd[{:.3f}].png'.format(args.problem, batch_id, bpd)))



        x = x_original
        x = x + step * mask.float() * torch.randn(x_grad.size()).to(args.device)   # random noise
        bpd, x_grad = f(x)
        print('gradient bpd: {:.4f}'.format(bpd))
        utils.save_image(rescaling_inv(x),
                         os.path.join(save_dir,
                                      'pixelcnn_{}_randnoise_mask{}_bpd[{:.3f}].png'.format(args.problem, batch_id, bpd)))
        # diff = x - x_original
        # diff *= 1000
        # utils.save_image(diff,
        #                  os.path.join(save_dir, 'pixelcnn_{}_mask_noise_{}.png'.
        #                               format(args.problem, batch_id)), normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('--problem', type=str,
                        default='mnist', help='Can be either cifar|mnist')
    # parser.add_argument('-p', '--print_every', type=int, default=50,
    #                     help='how many iterations between print statements')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Every how many epochs to write checkpoint/samples?')

    # model
    parser.add_argument('--nr_resnet', type=int, default=2,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('--nr_filters', type=int, default=60,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('--nr_logistic_mix', type=int, default=3,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size during training per GPU')
    parser.add_argument('--epochs', type=int,
                        default=10, help='How many epochs to run in total?')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed to use')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--n_classes", type=int,
                        default=10, help="number of classes of dataset.")

    parser.add_argument("--translation_attack", action="store_true",
                        help="perform translation attack")
    parser.add_argument("--gradient_attack", action="store_true",
                        help="perform gradient attack")

    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")

    args.sample_batch_size = 25
    args.obs = (1, 28, 28) if args.problem == 'mnist' or args.problem == 'fashion' else (3, 32, 32)
    input_channels = args.obs[0]

    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                     input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    model = model.to(args.device)

    if args.translation_attack:
        translation_attack(model, args)
    elif args.gradient_attack:
        gradient_attack(model, args)
    else:
        train(model, args)

