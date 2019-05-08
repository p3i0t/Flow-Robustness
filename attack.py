import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from NormalizingFlows.pixelcnn_utils import discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d
from NormalizingFlows.pixelcnn import PixelCNN
from NormalizingFlows import ConGlow


# Pre & Post processing functions for Glow
def preprocess(x):
    x = x * (args.n_bins - 1)
    x = x / args.n_bins - 0.5
    return x


def postprocess(x):
    return (x + 0.5).clamp(0., 1.)

# Pre & Post processing for PixelCNN
rescaling = lambda x: (x - .5) * 2.
rescaling_inv = lambda x: .5 * x + .5


def get_dataset_ops(args):
    print('image size ', args.image_size)
    if args.image_size != 28:
        ds_transforms = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                            transforms.ToTensor()])
    else:
        ds_transforms = transforms.ToTensor()

    if args.problem == 'mnist':
        dir = os.path.join(args.data_dir, 'MNIST')
        test_set = datasets.MNIST(dir, download=True, train=False, transform=ds_transforms)
    elif args.problem == 'fashion':
        dir = os.path.join(args.data_dir, 'FashionMNIST')
        test_set = datasets.FashionMNIST(dir, download=True, train=False, transform=ds_transforms)

    else:
        raise Exception('{} dataset not in [mnist, fashion]'.format(args.dataset))

    loss_op = lambda real, fake: discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x: sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

    return test_set, loss_op, sample_op


def plot_quality_image(x, file_name):
    return
    trans = transforms.ToPILImage(mode='L')
    plt.imshow(trans(x[0].cpu()))

    plt.tight_layout()
    plt.gca().xaxis.set_ticks([])
    plt.gca().yaxis.set_ticks([])
    plt.savefig(file_name, dpi=200, pad_inches=0, bbox_inches='tight')
    plt.clf()


def left_shift(x, n_pixel=1):
    return torch.cat([x[:, :, :, n_pixel:], x[:, :, :, :n_pixel]], dim=-1)


def load_glow(glow, args):
    # Load pretrained Glow
    checkpoint = torch.load(os.path.join(args.log_dir, '{}_glow_{}.pth'.format(args.coupling, args.problem))
                            , map_location=lambda storage, loc: storage)
    glow.load_state_dict(checkpoint['model_state'])
    return glow


def load_pixel_cnn(pixel_cnn, args):
    # Load pretrained PixelCNN
    pixel_cnn_name = 'pixelcnn_{}_{}'.format(args.problem, args.image_size)
    check_point = torch.load(os.path.join(args.log_dir, '{}.pth'.format(pixel_cnn_name)),
                             map_location=lambda storage, loc: storage)
    pixel_cnn.load_state_dict(check_point['state_dict'])
    return pixel_cnn


def pixelcnn_translation_attack(pixel_cnn, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pixel_cnn = load_pixel_cnn(pixel_cnn, args)
    pixel_cnn.eval()

    # Dataloader
    args.batch_size = 1
    args.image_size = 28
    args.obs = (1, args.image_size, args.image_size)
    test_set, loss_op, sample_op = get_dataset_ops(args)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)  # batch size fixed to 1

    # wrapper function of PixelCNN
    def pixel_cnn_f(x):
        x = rescaling(x)
        deno = args.batch_size * np.prod(args.obs) * np.log(2.)
        output = pixel_cnn(x)
        loss = loss_op(x, output)
        bpd = loss / deno
        return bpd.item()

    # Create directory
    save_dir = os.path.join(args.save_dir, 'translation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bpd_list = []
    with torch.no_grad():
        n_samples = args.n_samples
        for sample_id, (x, y) in enumerate(test_loader):
            if sample_id == n_samples:
                break

            x = x.to(args.device)
            y = y.to(args.device)

            bpd_0 = pixel_cnn_f(x)
            plot_quality_image(x, os.path.join(save_dir, 'pcnn_{}_original_{}.png'.format(args.problem, sample_id)))

            x = left_shift(x, n_pixel=1)
            bpd_1 = pixel_cnn_f(x)
            plot_quality_image(x, os.path.join(save_dir, 'pcnn_{}_l1_{}.png'.format(args.problem, sample_id)))

            x = left_shift(x, n_pixel=1)
            bpd_2 = pixel_cnn_f(x)
            plot_quality_image(x, os.path.join(save_dir, 'pcnn_{}_l2_{}.png'.format(args.problem, sample_id)))

            bpd_list.append((bpd_0, bpd_1, bpd_2))

    return bpd_list


def glow_translation_attack(glow, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    glow = load_glow(glow, args)

    glow.eval()

    # Dataloader
    args.batch_size = 1
    args.image_size = 32
    test_set, loss_op, sample_op = get_dataset_ops(args)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)  # batch size fixed to 1

    # wrapper function of Glow
    def glow_f(x, y):
        loglikelihood = torch.zeros(x.size(0)).to(args.device)

        n_pixels = np.prod(x.size()[1:])
        loglikelihood += -np.log(args.n_bins) * n_pixels
        z, loglikelihood, eps_list = glow(preprocess(x), loglikelihood, y)

        bpd = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
        return bpd.item()

    # Create directory
    save_dir = os.path.join(args.save_dir, 'translation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bpd_list = []
    # Main part
    with torch.no_grad():
        n_samples = args.n_samples
        for sample_id, (x, y) in enumerate(test_loader):
            if sample_id == n_samples:
                break

            x = x.to(args.device)
            y = y.to(args.device)

            bpd_0 = glow_f(x, y)
            plot_quality_image(x, os.path.join(save_dir, 'glow_{}_original_{}.png'.format(args.problem, sample_id)))

            x = left_shift(x, n_pixel=1)
            bpd_1 = glow_f(x, y)
            plot_quality_image(x, os.path.join(save_dir, 'glow_{}_l1_{}.png'.format(args.problem, sample_id)))

            x = left_shift(x, n_pixel=1)
            bpd_2 = glow_f(x, y)
            plot_quality_image(x, os.path.join(save_dir, 'glow_{}_l2_{}.png'.format(args.problem, sample_id)))

            bpd_list.append((bpd_0, bpd_1, bpd_2))

    return bpd_list


def translation_attack(pixel_cnn, glow, args):
    pixelcnn_bpd_list = pixelcnn_translation_attack(pixel_cnn, args)
    glow_bpd_list = glow_translation_attack(glow, args)
    for i in range(len(pixelcnn_bpd_list)):
        print('=====>Sample {}<===='.format(i + 1))
        print('Original image BPDs, PixelCNN: {:.3f}, Glow: {:.3f}'.format(pixelcnn_bpd_list[i][0], glow_bpd_list[i][0]))
        print('1p-left image BPDs, PixelCNN: {:.3f}, Glow: {:.3f}'.format(pixelcnn_bpd_list[i][1], glow_bpd_list[i][1]))
        print('2p-left image BPDs, PixelCNN: {:.3f}, Glow: {:.3f}'.format(pixelcnn_bpd_list[i][2], glow_bpd_list[i][2]))


def perturbation_attack(pixel_cnn, glow, args):
    pixel_cnn.eval()
    glow.eval()
    torch.manual_seed(args.seed+100)
    np.random.seed(args.seed+100)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load pretrained PixelCNN
    pixel_cnn_name = 'pixelcnn_{}_{}'.format(args.problem, args.image_size)
    check_point = torch.load(os.path.join(args.log_dir, '{}.pth'.format(pixel_cnn_name)),
                             map_location=lambda storage, loc: storage)
    pixel_cnn.load_state_dict(check_point['state_dict'])

    # Load pretrained Glow
    checkpoint = torch.load(os.path.join(args.log_dir, '{}_glow_{}.pth'.format(args.coupling, args.problem))
                            , map_location=lambda storage, loc: storage)
    glow.load_state_dict(checkpoint['model_state'])

    # Dataloader
    args.batch_size = 1
    test_set, loss_op, sample_op = get_dataset_ops(args)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)  # batch size fixed to 1

    # Create directory
    save_dir = os.path.join(args.save_dir, 'perturbation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # wrapper function of PixelCNN
    def pixel_cnn_f(x):
        deno = args.batch_size * np.prod(args.obs) * np.log(2.)
        output = pixel_cnn(x)
        loss = loss_op(x, output)
        bpd = loss / deno
        return bpd.item()

    # wrapper function of Glow
    def glow_f(x, y):
        loglikelihood = torch.zeros(x.size(0)).to(args.device)

        n_pixels = np.prod(x.size()[1:])
        loglikelihood += -np.log(args.n_bins) * n_pixels
        z, loglikelihood, eps_list = glow(x, loglikelihood, y)

        bpd = (- loglikelihood) / (np.log(2.) * n_pixels)  # bits per pixel
        return bpd.item()

    n_samples = 3
    for sample_id, (x, y) in enumerate(test_loader):
        if sample_id == n_samples:
            break

        print('=====>Sample {}<====='.format(sample_id + 1))

        mask = (x < 0.3).to(args.device)

        x = x.to(args.device)
        y = y.to(args.device)

        # Original image
        pixel_cnn_bpd = pixel_cnn_f(rescaling(x))
        glow_bpd = glow_f(preprocess(x), y)
        print('Original image BPDs, PixelCNN: {:.3f}, Glow: {:.3f}'.format(pixel_cnn_bpd, glow_bpd))
        plot_quality_image(x, os.path.join(save_dir, '{}_original_{}.png'.format(args.problem, sample_id)))

        # Add random gaussian noise
        eps = 1e-3
        noise = eps * torch.randn(x.size()).to(args.device)
        pixelcnn_x_perturb = rescaling(x) + noise

        pixel_cnn_bpd = pixel_cnn_f(pixelcnn_x_perturb)
        glow_bpd = glow_f(preprocess(x) + noise, y)

        print('Image  BPDs (random noise), PixelCNN: {:.3f}, Glow: {:.3f}'.format(pixel_cnn_bpd, glow_bpd))
        plot_quality_image(rescaling_inv(pixelcnn_x_perturb),
                           os.path.join(save_dir, '{}_noise_img_{}.png'.format(args.problem, sample_id)))

        clamp_noise = torch.clamp(noise * 1e3, -1., 1.)
        plot_quality_image(clamp_noise,  os.path.join(save_dir, '{}_noise_{}.png'.format(args.problem, sample_id)))

        # Add masked random gaussian noise
        mask_noise = eps * mask.float() * torch.randn(x.size()).to(args.device)
        pixelcnn_mask_x_perturb = rescaling(x) + mask_noise
        pixel_cnn_bpd = pixel_cnn_f(pixelcnn_mask_x_perturb)
        glow_bpd = glow_f(preprocess(x) + mask_noise, y)
        print('Image  BPDs (masked random noise), PixelCNN: {:.3f}, Glow: {:.3f}'.format(pixel_cnn_bpd, glow_bpd))
        plot_quality_image(rescaling_inv(pixelcnn_mask_x_perturb),
                           os.path.join(save_dir, '{}_mask_noise_img_{}.png'.format(args.problem, sample_id)))

        clamp_mask_noise = torch.clamp(mask_noise * 1e3, -1., 1.)
        plot_quality_image(clamp_mask_noise, os.path.join(save_dir, '{}_mask_noise_{}.png'.format(args.problem, sample_id)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('--save_dir', type=str, default='logs_attack',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('--problem', type=str,
                        default='mnist', help='Can be either cifar|mnist')

    # PixelCNN++
    parser.add_argument('--nr_resnet', type=int, default=2,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('--nr_filters', type=int, default=100,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('--nr_logistic_mix', type=int, default=3,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument("--image_size", type=int,
                        default=32, help="Image size")


    # Glow
    parser.add_argument("--width", type=int, default=128,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=8,
                        help="Depth of network")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=5,
                        help="Number of levels")
    parser.add_argument("--permutation", type=str, default='conv1x1',
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--coupling", type=str, default='affine',
                        help="Coupling type: 0=additive, 1=affine")
    parser.add_argument("--learn_top", action="store_true",
                        help="Learn spatial prior")

    # Hyperparameters
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed to use')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--n_classes", type=int,
                        default=10, help="number of classes of dataset.")
    parser.add_argument("--n_samples", type=int,
                        default=3, help="number of classes of dataset.")

    # Attack options
    parser.add_argument("--translation_attack", action="store_true",
                        help="perform translation attack")
    parser.add_argument("--perturbation_attack", action="store_true",
                        help="perform gradient attack")

    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")

    # Create PixelCNN
    args.obs = (1, args.image_size, args.image_size)

    pixel_cnn = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                         input_channels=1, nr_logistic_mix=args.nr_logistic_mix)
    pixel_cnn = pixel_cnn.to(args.device)

    # Create Glow
    args.n_bins = 2. ** args.n_bits_x  # number of pixel levels

    args.in_channels = 1 if args.problem == 'mnist' or args.problem == 'fashion' else 3
    args.hidden_channels = args.width

    glow = ConGlow(args).to(args.device)

    if args.translation_attack:
        translation_attack(pixel_cnn, glow, args)
    elif args.perturbation_attack:
        perturbation_attack(pixel_cnn, glow, args)
    else:
        print('Choose attack type')
