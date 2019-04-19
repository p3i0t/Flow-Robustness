import torch


def mean(x, dim=None, keepdim=False):
    if dim is None:
        return x.mean()  # mean of all
    else:
        if isinstance(dim, int):
            return x.mean(dim, keepdim)
        else:
            assert isinstance(dim, list)
            dim = sorted(dim, reverse=True)
            for d in dim:
                x = x.mean(dim=d, keepdim=keepdim)

            return x


def mean_test():
    x = torch.Tensor(range(32)).view(1, 4, 2, 4)
    m = mean(x, dim=[1, 3], keepdim=True)
    print(m)

    m = x.mean(dim=3).mean(dim=1)
    print(m)

if __name__ == '__main__':
    mean_test()
