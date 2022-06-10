# This script gets the mean and std deviation of each channel in the image datasets
# To normalise them as a preproc method

import os
import torch
from torchvision import datasets
from torchvision import transforms

# specify folder and selected datasets
base_folder = "dataset"

# dataset = "SVHN"
dataset = "FashionMNIST"
# dataset = "MNIST"
# dataset = "CIFAR"

if dataset == "SVHN":
    torch_dataset = datasets.SVHN(
        os.path.join(base_folder, "data-svhn"),
        split="train",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
elif dataset == "FashionMNIST":
    torch_dataset = datasets.FashionMNIST(
        os.path.join(base_folder, "data-fashion-mnist"),
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
elif dataset == "MNIST":
    torch_dataset = datasets.MNIST(
        os.path.join(base_folder, "data-mnist"),
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
elif dataset == "CIFAR":
    torch_dataset = datasets.CIFAR10(
        os.path.join(base_folder, "data-cifar"),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.4914008, 0.482159, 0.44653094],
                #     std=[0.24703224, 0.24348514, 0.26158786],
                # ),
            ]
        ),
    )

# prepare data loader
train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=len(torch_dataset))

# get whole data
all_data = next(iter(train_loader))[0]

# channel axis
channel_axis = 1

# get mean and std across axes
dims = [i for i in range(len(all_data.shape)) if i != channel_axis]
mean_channels = all_data.mean(dim=dims)
std_channels = all_data.std(dim=dims)

print("DATA SHAPE:" + str(all_data.shape))
print("DATASET:" + str(dataset))
print("MEAN:" + str(list(mean_channels.numpy())))
print("STD:" + str(list(std_channels.numpy())))

## RESULTS:
## DATASET:CIFAR
## MEAN:[0.4914008, 0.482159, 0.44653094]
## STD:[0.24703224, 0.24348514, 0.26158786]

## DATASET:SVHN
## MEAN:[0.43768448, 0.44376868, 0.4728041]
## STD:[0.19803013, 0.20101564, 0.19703615]

## DATASET:FashionMNIST
## MEAN:[0.2860406]
## STD:[0.35302424]

## DATASET:MNIST
## MEAN:[0.13066047]
## STD:[0.30810782]
