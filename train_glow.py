import argparse
import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim import Adam, Adamax

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


def train(glow, optimizer, hps):
    glow.train()
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
            z, loglikelihood, eps_list = glow(x, loglikelihood, y)

            # Generative loss
            bits_x = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
            mean_bits_x = bits_x.mean()
            mean_bits_x.backward()
            optimizer.step()

            bits_list.append(mean_bits_x.cpu().item())

        # sampling images.
        save_image(postprocess(x), os.path.join(hps.log_dir, 'glow_epoch{}_original.png'.format(epoch)))
        x_reverse = glow.reverse(eps_list, y)
        save_image(postprocess(x_reverse), os.path.join(hps.log_dir, 'glow_epoch{}_reverse.png'.format(epoch)))

        temperatures = [0., 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]

        sample_labels = torch.arange(0, 100).long().to(hps.device)/10
        for temp_id, temp in enumerate(temperatures):
            sample = glow.sample(sample_labels, temp)
            save_image(postprocess(sample), os.path.join(hps.log_dir, 'epoch{}_sample_{}.png'.format(epoch, temp_id)))

        cur_bits_per_dim = np.mean(bits_list)
        print('Epoch {}, mean bits_per_dim: {:.4f}'.format(epoch, cur_bits_per_dim))

        if cur_bits_per_dim < best_bits_per_dim:
            best_bits_per_dim = cur_bits_per_dim
            checkpoint = {'model_state': glow.state_dict(),
                          'bits_per_dim': best_bits_per_dim,
                          'hps': hps
                          }
            suffix = '' if hps.class_id == -1 else '_{}'.format(hps.class_id)
            torch.save(checkpoint, os.path.join(hps.log_dir, '{}_glow_{}{}.pth'.format(hps.coupling, hps.problem, suffix)))
            print('==> New optimal model saved !!!')


def inference(glow, hps):
    glow.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    suffix = '' if hps.class_id == -1 else '_{}'.format(hps.class_id)
    checkpoint = torch.load(os.path.join('logs/', 'glow_{}{}.pth'.format(hps.problem, suffix))
                            , map_location = lambda storage, loc: storage)
    glow.load_state_dict(checkpoint['model_state'])

    dataset = get_dataset(dataset=hps.infer_problem, train=False, class_id=hps.infer_class_id)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=True)

    bits_list = []

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(test_loader):
            x = preprocess(x).to(hps.device)
            x = x + torch.empty(x.size()).uniform_(0, 1 / hps.n_bins).to(hps.device)  # add small uniform noise

            y = y.to(hps.device)
            loglikelihood = torch.zeros(x.size(0)).to(hps.device)

            n_pixels = np.prod(x.size()[1:])
            loglikelihood += -np.log(hps.n_bins) * n_pixels

            z, loglikelihood, eps_list = glow(x, loglikelihood, y)

            # Generative loss
            bits_x = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
            mean_bits_x = bits_x.mean()

            bits_list.append(mean_bits_x.cpu().item())

    cur_bits_per_dim = np.mean(bits_list)
    desc = 'all classes' if hps.class_id == -1 else 'class {}'.format(hps.infer_class_id)
    print('Inference on {}, mean bits_per_dim: {:.4f}'.format(desc, cur_bits_per_dim))


def translation_attack(glow, hps):
    glow.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    suffix = '' if hps.class_id == -1 else '_{}'.format(hps.class_id)
    checkpoint = torch.load(os.path.join(hps.log_dir, '{}_glow_{}{}.pth'.format(hps.coupling, hps.problem, suffix))
                            , map_location=lambda storage, loc: storage)
    glow.load_state_dict(checkpoint['model_state'])

    dataset = get_dataset(dataset=hps.problem, train=True, class_id=hps.class_id)
    train_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)
    test_set = get_dataset(dataset=hps.problem, train=False, class_id=hps.class_id)
    test_loader = DataLoader(dataset=test_set, batch_size=hps.n_batch_test, shuffle=False)
    # infer_dataset = get_dataset(dataset=hps.infer_problem, train=False, class_id=hps.infer_class_id)
    # infer_loader = DataLoader(dataset=infer_dataset, batch_size=hps.n_batch_test, shuffle=False)

    save_dir = os.path.join(hps.log_dir, 'translation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def f(x, y):
        loglikelihood = torch.zeros(x.size(0)).to(hps.device)

        n_pixels = np.prod(x.size()[1:])
        loglikelihood += -np.log(hps.n_bins) * n_pixels
        z, loglikelihood, eps_list = glow(x, loglikelihood, y)

        bits_x = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
        return bits_x, eps_list

    def left_shift(x, n_pixel=1):
        return torch.cat([x[:, :, :, n_pixel:], x[:, :, :, :n_pixel]], dim=-1)

    def up_shift(x, n_pixel=1):
        return torch.cat([x[:, :, n_pixel:, :], x[:, :, :n_pixel, :]], dim=-2)

    def left_up(x, n_pixel=1):
        x = left_shift(x, n_pixel)
        x = up_shift(x, n_pixel)
        return x

    def eval_bits(data_loader, hps, left_pixel=None):
        pix = 0 if left_pixel is None else left_pixel
        keys = ['{}_class{}_leftpixel{}'.format(hps.problem, i, pix) for i in range(hps.n_classes)]
        bits_dict = {key: list() for key in keys}

        for batch_id, (x, y) in enumerate(data_loader):
            x = preprocess(x).to(hps.device)
            y = y.to(hps.device)
            if left_pixel:
                x = left_shift(x, left_pixel)
                if hps.problem == 'mnist':
                    x = up_shift(x, left_pixel)
            bits_x, _ = f(x, y)
            for i in range(hps.n_classes):
                bits_dict['{}_class{}_leftpixel{}'.format(hps.problem, i, pix)] += list(bits_x[y == i].cpu().numpy())
        return bits_dict

    with torch.no_grad():
        # Evaluate on test set with different pixel shifts to left
        bits_dict = {}
        bits = eval_bits(test_loader, hps)
        bits_dict.update(bits)

        left_bits_1 = eval_bits(test_loader, hps, left_pixel=1)
        left_bits_2 = eval_bits(test_loader, hps, left_pixel=2)
        bits_dict.update(left_bits_1)
        bits_dict.update(left_bits_2)

        torch.save(bits_dict, os.path.join(save_dir, 'glow_{}_bits_dict.pth'.format(hps.problem)))

        # Generate some samples.
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
        n_samples = 1

        for sample_id, (x, y) in enumerate(test_loader):
            if sample_id == n_samples:
                break
            x = preprocess(x).to(hps.device)
            y = y.to(hps.device)

            bits_x, _ = f(x, y)
            save_image(postprocess(x), os.path.join(save_dir, '{}_original_{}_bpd[{:.4f}].png'.format(
                hps.problem, sample_id, bits_x.cpu().item())))

            x = left_shift(x, n_pixel=1)
            if hps.problem == 'mnist':
                x = up_shift(x, n_pixel=1)
            bits_x, _ = f(x, y)
            save_image(postprocess(x), os.path.join(save_dir, '{}_l1_{}_bpd[{:.4f}].png'.format(
                hps.problem, sample_id, bits_x.cpu().item())))

            x = left_shift(x, n_pixel=1)
            if hps.problem == 'mnist':
                x = up_shift(x, n_pixel=1)
            bits_x, _ = f(x, y)
            save_image(postprocess(x), os.path.join(save_dir, '{}_l2_{}_bpd[{:.4f}].png'.format(
                hps.problem, sample_id, bits_x.cpu().item())))


def reverse_attack(glow, hps):
    glow.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    suffix = '' if hps.class_id == -1 else '_{}'.format(hps.class_id)
    checkpoint = torch.load(os.path.join(hps.log_dir, '{}_glow_{}{}.pth'.format(hps.coupling, hps.problem, suffix))
                            , map_location=lambda storage, loc: storage)
    glow.load_state_dict(checkpoint['model_state'])

    test_set = get_dataset(dataset=hps.problem, train=False, class_id=hps.class_id)
    test_loader = DataLoader(dataset=test_set, batch_size=hps.n_batch_test, shuffle=False)
    # infer_dataset = get_dataset(dataset=hps.infer_problem, train=False, class_id=hps.infer_class_id)
    # infer_loader = DataLoader(dataset=infer_dataset, batch_size=hps.n_batch_test, shuffle=False)

    save_dir = os.path.join(hps.log_dir, 'reverse')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def f(x, y):
        loglikelihood = torch.zeros(x.size(0)).to(hps.device)

        n_pixels = np.prod(x.size()[1:])
        loglikelihood += -np.log(hps.n_bins) * n_pixels
        z, loglikelihood, eps_list = glow(x, loglikelihood, y)

        bits_x = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
        return bits_x, eps_list

    def zero_epses(eps_list, n_zeros=1):
        "Zero the preceding n_zeros eps factors. "
        assert n_zeros < len(eps_list)
        for idx in range(n_zeros):
            eps_list[idx].zero_()
        return eps_list

    def eval_bits(data_loader, hps):
        keys = ['{}_class{}_zero{}'.format(hps.problem, i, j) for i in range(hps.n_classes) for j in range(3)]
        bits_dict = {key: list() for key in keys}

        def remove_nan(x):
            return x[~np.isnan(x)]

        for batch_id, (x, y) in enumerate(data_loader):
            x = preprocess(x).to(hps.device)
            y = y.to(hps.device)
            bits, eps_list = f(x, y)
            for i in range(hps.n_classes):
                bits_dict['{}_class{}_zero0'.format(hps.problem, i)] += list(bits[y == i].cpu().numpy())

            zeroed_eps_list = zero_epses(eps_list, n_zeros=1)
            x_reverse = glow.reverse(zeroed_eps_list, y)
            reverse_bits, eps_list = f(x_reverse, y)
            for i in range(hps.n_classes):
                bits_ = remove_nan(reverse_bits[y == i].cpu().numpy())
                bits_dict['{}_class{}_zero1'.format(hps.problem, i)] += list(bits_)

            zeroed_eps_list = zero_epses(eps_list, n_zeros=2)
            x_reverse = glow.reverse(zeroed_eps_list, y)
            reverse_bits, _ = f(x_reverse, y)
            for i in range(hps.n_classes):
                bits_ = remove_nan(reverse_bits[y == i].cpu().numpy())
                bits_dict['{}_class{}_zero2'.format(hps.problem, i)] += list(bits_)

        return bits_dict

    with torch.no_grad():
        # bits_dict = eval_bits(test_loader, hps)
        # torch.save(bits_dict, os.path.join(save_dir, 'glow_{}_bits_dict.pth'.format(hps.problem)))

        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

        n_samples = 5

        transfer_dict = {}
        for batch_id, (x, y) in enumerate(test_loader):
            if batch_id == n_samples:
                break

            x = preprocess(x).to(hps.device)
            y = y.to(hps.device)
            bits, eps_list = f(x, y)

            save_image(postprocess(x), os.path.join(save_dir, '{}_original_{}_bpd[{:.4f}].png'.format(
                hps.problem, batch_id, bits.cpu().item())))

            zeroed_eps_list = zero_epses(eps_list, n_zeros=1)
            x_reverse = glow.reverse(zeroed_eps_list, y)
            reverse_bits, eps_list = f(x_reverse, y)

            save_image(postprocess(x_reverse), os.path.join(save_dir, '{}_zero1_{}_bpd[{:.4f}].png'.format(
                hps.problem, batch_id, reverse_bits.cpu().item())))

            zeroed_eps_list = zero_epses(eps_list, n_zeros=2)
            x_reverse_ = glow.reverse(zeroed_eps_list, y)
            reverse_bits_, _ = f(x_reverse_, y)
            save_image(postprocess(x_reverse_), os.path.join(save_dir, '{}_zero2_{}_bpd[{:.4f}].png'.format(
                hps.problem, batch_id, reverse_bits_.cpu().item())))

            transfer_x = torch.cat([x, x_reverse, x_reverse_], dim=1)
            transfer_y = torch.cat([bits, reverse_bits, reverse_bits_])
            transfer_dict['x{}'.format(batch_id)] = transfer_x
            transfer_dict['y{}'.format(batch_id)] = transfer_y

        torch.save(transfer_dict, os.path.join(save_dir, 'transfer_xy.pth'))

    # # out-distribution evaluation
    # in_class_id = hps.infer_class_id
    # in_set = get_dataset(dataset='fashion', train=False, class_id=in_class_id)
    # in_loader = DataLoader(dataset=in_set, batch_size=hps.n_batch_test, shuffle=False)
    # out_set = get_dataset(dataset='mnist', train=False, class_id=-1)
    # out_loader = DataLoader(dataset=out_set, batch_size=hps.n_batch_test, shuffle=False)

    # fixed_y = None
    # in_bits_list = []
    # for batch_id, (x, y) in enumerate(in_loader):
    #     x = preprocess(x).to(hps.device)
    #     y = y.to(hps.device)
    #     if batch_id == 0:
    #         fixed_y = y
    #     bits, eps_list = f(x, y)
    #     in_bits_list += list(bits.cpu().detach().numpy())
    #
    # print('in_bits: ', np.mean(in_bits_list))
    #
    # out_bits_list = []
    # reverse_out_bits_list = []
    # for batch_id, (x, y) in enumerate(out_loader):
    #     x = preprocess(x).to(hps.device)
    #     # y = y.to(hps.device)
    #     bits, eps_list = f(x, fixed_y)
    #     out_bits_list += list(bits.cpu().detach().numpy())
    #
    #     zeroed_eps_list = zero_epses(eps_list, n_zeros=2)
    #     x_reverse = glow.reverse(zeroed_eps_list, fixed_y)
    #     reverse_bits, _ = f(x_reverse, fixed_y)
    #     reverse_out_bits_list += list(reverse_bits.cpu().detach().numpy())
    #
    # print('out_bits: ', np.mean(out_bits_list))
    # print('reverse_out_bits: ', np.mean(reverse_out_bits_list))
    # bits_dict = {
    #              'in_bits': in_bits_list,
    #              'out_bits': out_bits_list,
    #              'zeroed_out_bits': reverse_out_bits_list,
    #              }
    # suffix = '_{}'.format(in_class_id)
    # torch.save(bits_dict, 'logs/glow_out_evaluation{}_attack2.pth'.format(suffix))

    # in_loader = DataLoader(dataset=in_set, batch_size=1, shuffle=False)
    # out_loader = DataLoader(dataset=out_set, batch_size=1, shuffle=False)
    #
    # def sample(loader, mode='out'):
    #     n_samples = 5
    #     for sample_id, (x, y) in enumerate(loader):
    #         if sample_id == n_samples:
    #             break
    #         x = preprocess(x).to(hps.device)
    #         #y = y.to(hps.device)
    #         y = torch.tensor([in_class_id]).long().to(hps.device)
    #
    #         bits_x, eps_list = f(x, y)
    #         save_image(postprocess(x), os.path.join('logs/', 'out_evaluation_in{}_{}sample{}_original_bpd[{:.4f}].png'.format(
    #             in_class_id, mode, sample_id, bits_x.cpu().item())))
    #
    #         eps_list_1 = zero_epses(eps_list, n_zeros=1)
    #         x_reverse_1 = glow.reverse(eps_list_1, y)
    #         bits_x_1, _ = f(x_reverse_1, y)
    #         save_image(postprocess(x_reverse_1),
    #                    os.path.join('logs/', 'out_evaluation_in{}_{}sample{}_zero1_bpd[{:.4f}].png'.format(
    #                        in_class_id, mode, sample_id, bits_x_1.cpu().item())))
    #
    #         bits_x, eps_list = f(x, y)
    #         eps_list_2 = zero_epses(eps_list, n_zeros=2)
    #         x_reverse_2 = glow.reverse(eps_list_2, y)
    #         bits_x_2, _ = f(x_reverse_2, y)
    #         save_image(postprocess(x_reverse_2),
    #                    os.path.join('logs/', 'out_evaluation_in{}_{}sample{}_zero2_bpd[{:.4f}].png'.format(
    #                        in_class_id, mode, sample_id, bits_x_2.cpu().item())))
    #
    # sample(in_loader, mode='in')
    # sample(out_loader, mode='out')


def gradient_attack(glow, hps):
    glow.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    suffix = '' if hps.class_id == -1 else '_{}'.format(hps.class_id)
    checkpoint = torch.load(os.path.join('logs/', 'glow_{}{}.pth'.format(hps.problem, suffix))
                            , map_location=lambda storage, loc: storage)
    glow.load_state_dict(checkpoint['model_state'])

    in_set = get_dataset(dataset=hps.problem, train=False, class_id=-1)
    in_loader = DataLoader(dataset=in_set, batch_size=1, shuffle=False)

    save_dir = os.path.join(hps.log_dir, 'gradient')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def f(x, y):
        loglikelihood = torch.zeros(x.size(0)).to(hps.device)

        n_pixels = np.prod(x.size()[1:])
        loglikelihood += -np.log(hps.n_bins) * n_pixels

        # optimizer.zero_grad()
        glow.zero_grad()
        z, loglikelihood, eps_list = glow(x, loglikelihood, y)

        # Generative loss
        bits_x = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
        grad = torch.autograd.grad(outputs=bits_x,
                                   inputs=x,
                                   grad_outputs=torch.ones(bits_x.size()).to(hps.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        return bits_x.item(), grad

    n_iterations = 6
    step = 1e-4

    mode = 'ascent'
    n_samples = 1

    for batch_id, (x, y) in enumerate(in_loader):
        if batch_id == n_samples:
            break
        mask = (x < 0.5).to(hps.device)
        x = preprocess(x).to(hps.device)
        x.requires_grad = True
        y = y.to(hps.device)

        x_original = x

        for i in range(n_iterations):
            bpd, x_grad = f(x, y)
            if i == 0:
                print('initial gradient bpd: {:.4f}'.format(bpd))
                save_image(postprocess(x),
                           os.path.join('logs/',
                                        'gradient_{}_original_{}_bpd[{:.2f}].png'.format(hps.problem, batch_id, bpd)))
            if mode == 'descent':
                x = x - step * x_grad
            elif mode == 'ascent':
                x = x + step * x_grad

        print('gradient bpd: {:.4f}'.format(bpd))
        save_image(postprocess(x),
                   os.path.join('logs/', 'gradient_{}_{}_{}_bpd[{:.2f}].png'.format(mode, hps.problem, batch_id, bpd)))

        diff = x - x_original
        diff *= 1000
        save_image(diff, os.path.join('logs/', 'gradient_{}_{}_noise_{}.png'.
                                      format(mode, hps.problem, batch_id)), normalize=True)

        x = x_original
        for i in range(n_iterations):
            bpd, x_grad = f(x, y)
            if mode == 'descent':
                x = x - step * mask.float() * x_grad
            elif mode == 'ascent':
                x = x + step * mask.float() * x_grad

        print('gradient mask bpd: {:.4f}'.format(bpd))
        save_image(postprocess(x),
                   os.path.join('logs/', 'gradient_mask_{}_{}_{}_bpd[{:.2f}].png'.format(mode, hps.problem, batch_id, bpd)))

        diff = x - x_original
        diff *= 1000
        save_image(diff, os.path.join('logs/', 'gradient_mask_{}_{}_noise_{}.png'.
                                      format(mode, hps.problem, batch_id)), normalize=True)


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
    parser.add_argument("--epochs", type=int, default=10,
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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

    glow = ConGlow(hps).to(hps.device)
    optimizer = Adam(glow.parameters(), lr=hps.lr)

    if hps.inference:
        inference(glow, hps)
    elif hps.translation_attack:
        translation_attack(glow, hps)
    elif hps.reverse_attack:
        reverse_attack(glow, hps)
    elif hps.gradient_attack:
        gradient_attack(glow, hps)
    else:
        train(glow, optimizer, hps)

