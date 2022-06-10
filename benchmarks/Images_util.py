import torch
from torchvision import datasets, transforms
import os

from util.add_noise import AddGaussianNoise, AddUniformNoise

train_batch_size = 100
test_samples = 100


def get_standardise_vals(dataset):
    # for z_standardising
    mean_maps = {
        "CIFAR": [0.4914008, 0.482159, 0.44653094],
        "SVHN": [0.43768448, 0.44376868, 0.4728041],
        "FashionMNIST": [0.2860406],
        "MNIST": [0.13066047],
    }
    std_maps = {
        "CIFAR": [0.24703224, 0.24348514, 0.26158786],
        "SVHN": [0.19803013, 0.20101564, 0.19703615],
        "FashionMNIST": [0.35302424],
        "MNIST": [0.30810782],
    }
    return mean_maps[dataset], std_maps[dataset]


def get_ood_set(
    ood_dataset="SVHN",
    n_channels=-1,
    test_samples=100,
    shuffle=False,
    resize=None,
    base_folder="dataset",
    standardise=None,
    noise_scale=None,
    noise_type=None,
):
    # check path dataset
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    # prepare data transformation
    if (
        n_channels == 3 and (ood_dataset == "FashionMNIST" or ood_dataset == "MNIST")
    ) or (n_channels == 1 and (ood_dataset == "CIFAR" or ood_dataset == "SVHN")):
        if resize is not None:
            data_trans_ = [
                transforms.Grayscale(num_output_channels=n_channels),
                transforms.Resize(resize),
                transforms.ToTensor(),
            ]
        else:
            data_trans_ = [
                transforms.Grayscale(num_output_channels=n_channels),
                transforms.ToTensor(),
            ]
    else:
        # Data transformation
        data_trans_ = [transforms.ToTensor()]

    # apply standardising
    if standardise is not None:
        data_trans_ += [transforms.Normalize(standardise[0], standardise[1])]

    # add noise
    if (noise_scale is not None) and (noise_type is not None):
        accepted_types = ["normal", "uniform"]
        if noise_type == "normal":
            data_trans_ += [AddGaussianNoise(0.0, noise_scale)]
        elif noise_type == "uniform":
            data_trans_ += [AddUniformNoise(noise_scale)]
        else:
            raise NotImplemented("Accepted noise types are " + str(accepted_types))
    data_transform = transforms.Compose(data_trans_)

    # Load ood set
    if ood_dataset == "SVHN":
        ood_loader_ = torch.utils.data.DataLoader(
            datasets.SVHN(
                os.path.join(base_folder, "data-svhn"),
                split="test",
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    elif ood_dataset == "FashionMNIST":
        ood_loader_ = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                os.path.join(base_folder, "data-fashion-mnist"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    elif ood_dataset == "MNIST":
        ood_loader_ = torch.utils.data.DataLoader(
            datasets.MNIST(
                os.path.join(base_folder, "data-mnist"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    elif ood_dataset == "CIFAR":
        ood_loader_ = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                os.path.join(base_folder, "data-cifar"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    else:
        raise NotImplemented("OOD set can be CIFAR, FashionMNIST, MNIST or SVHN only.")
    return ood_loader_


def get_id_set(
    id_dataset="CIFAR",
    n_channels=-1,
    shuffle=True,
    train_batch_size=100,
    test_samples=100,
    resize=None,
    base_folder="dataset",
    standardise=None,
    noise_scale=None,
    noise_type=None,
):
    # check path dataset
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    # prepare data transformation
    if (
        n_channels == 3 and (id_dataset == "FashionMNIST" or id_dataset == "MNIST")
    ) or (n_channels == 1 and (id_dataset == "CIFAR" or id_dataset == "SVHN")):
        if resize is not None:
            data_trans_ = [
                transforms.Grayscale(num_output_channels=n_channels),
                transforms.Resize(resize),
                transforms.ToTensor(),
            ]
        else:
            data_trans_ = [
                transforms.Grayscale(num_output_channels=n_channels),
                transforms.ToTensor(),
            ]
    else:
        # Data transformation
        data_trans_ = [transforms.ToTensor()]

    # apply standardising
    if standardise is not None:
        data_trans_ += [transforms.Normalize(standardise[0], standardise[1])]

    # add noise
    if (noise_scale is not None) and (noise_type is not None):
        accepted_types = ["normal", "uniform"]
        if noise_type == "normal":
            data_trans_ += [AddGaussianNoise(0.0, noise_scale)]
        elif noise_type == "uniform":
            data_trans_ += [AddUniformNoise(noise_scale)]
        else:
            raise NotImplemented("Accepted noise types are " + str(accepted_types))
    data_transform = transforms.Compose(data_trans_)

    # Load ID set
    if id_dataset == "SVHN":
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                os.path.join(base_folder, "data-svhn"),
                split="train",
                download=True,
                transform=data_transform,
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                os.path.join(base_folder, "data-svhn"),
                split="test",
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=False,
        )
    elif id_dataset == "FashionMNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                os.path.join(base_folder, "data-fashion-mnist"),
                train=True,
                download=True,
                transform=data_transform,
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                os.path.join(base_folder, "data-fashion-mnist"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=False,
        )
    elif id_dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                os.path.join(base_folder, "data-mnist"),
                train=True,
                download=True,
                transform=data_transform,
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                os.path.join(base_folder, "data-mnist"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=False,
        )
    elif id_dataset == "CIFAR":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                os.path.join(base_folder, "data-cifar"),
                train=True,
                download=True,
                transform=data_transform,
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                os.path.join(base_folder, "data-cifar"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=False,
        )
    else:
        raise NotImplemented("ID set can be CIFAR, FashionMNIST, MNIST or SVHN only.")
    return train_loader, test_loader
