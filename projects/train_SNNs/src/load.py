#!/home/rj/.pyenv/shims/python

import torch
from torchvision import datasets, transforms

import tonic
import tonic.transforms as tonic_transforms

# ---------- default transform per dataset ----------#

default_transforms = {
        'mnist': transforms.ToTensor(),
        'fashion-mnist': transforms.ToTensor(),
        'cifar10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'n-mnist': tonic_transforms.Compose([
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=10),
        ]),
        'dvsgesture': tonic_transforms.Compose([
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, n_time_bins=10),
        ]),
        'shd': tonic_transforms.ToFrame(sensor_size=tonic.datasets.SHD.sensor_size, n_time_bins=10),
        'ssc': tonic_transforms.ToFrame(sensor_size=tonic.datasets.SSC.sensor_size, n_time_bins=10),
    }

available_datasets = list(default_transforms.keys()) + ['smnist', 'psmnist']

def load_dataset(name, root='./data', batch_size=128, transform=None):
    name = name.lower()


    # ---------- resolve transform ----------
    if name in ('smnist', 'psmnist'):
        if transform is not None:
            tf = transform
        else:
            base_tf = [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1, 1))]
            if name == 'psmnist':
                perm = torch.randperm(784)
                base_tf.append(transforms.Lambda(lambda x: x[perm]))
            tf = transforms.Compose(base_tf)
    else:
        tf = transform or default_transforms.get(name)

    # ---------- load dataset ----------
    match name:
        case 'mnist':
            train = datasets.MNIST(root, train=True, download=True, transform=tf)
            test = datasets.MNIST(root, train=False, download=True, transform=tf)

        case 'fashion-mnist':
            train = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
            test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)

        case 'cifar10':
            train = datasets.CIFAR10(root, train=True, download=True, transform=tf)
            test = datasets.CIFAR10(root, train=False, download=True, transform=tf)

        case 'smnist' | 'psmnist':
            train = datasets.MNIST(root, train=True, download=True, transform=tf)
            test = datasets.MNIST(root, train=False, download=True, transform=tf)

        case 'n-mnist':
            train = tonic.datasets.NMNIST(root, train=True, transform=tf)
            test = tonic.datasets.NMNIST(root, train=False, transform=tf)

        case 'dvsgesture':
            train = tonic.datasets.DVSGesture(root, train=True, transform=tf)
            test = tonic.datasets.DVSGesture(root, train=False, transform=tf)

        case 'shd':
            train = tonic.datasets.SHD(root, train=True, transform=tf)
            test = tonic.datasets.SHD(root, train=False, transform=tf)

        case 'ssc':
            train = tonic.datasets.SSC(root, split='train', transform=tf)
            test = tonic.datasets.SSC(root, split='test', transform=tf)

        case _:
            raise ValueError(f"Unknown dataset: {name}")

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    
    # avaliable datasets 
    print("Available datasets:", available_datasets)
    print("Loading Fashion-MNIST dataset...")
    train_loader, test_loader = load_dataset('fashion-mnist')
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)
