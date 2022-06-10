import numpy as np
import torch


def add_noise(*data, noise_type="normal", noise_scale=1, clip=True):
    """
    Adds additive noise to a set of data; either normal or uniform noise
    With different noise scales
    """
    noisy_data = data
    if noise_type == "normal":
        noisy_data = (dt + np.random.normal(0, noise_scale, dt.shape) for dt in data)
    elif noise_type == "uniform":
        noisy_data = (
            dt + np.random.uniform(-noise_scale, noise_scale, dt.shape) for dt in data
        )
    if clip:
        noisy_data = (np.clip(dt, 0, 1) for dt in noisy_data)
    return noisy_data


# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, clip=True):
        self.std = std
        self.mean = mean
        self.clip = clip

    def __call__(self, tensor):
        noisy_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        # noisy_tensor = tensor + torch.clip(
        #     torch.randn(tensor.size()) * self.std + self.mean, 0, 1
        # )
        # print(tensor.shape)
        # noisy_tensor = (
        #     tensor + torch.randn(1, 1, 1).repeat(3, 32, 32) * self.std + self.mean
        # )
        if self.clip:
            noisy_tensor = torch.clip(noisy_tensor, 0, 1)
        return noisy_tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AddUniformNoise(object):
    """
    # Adds U(+scale,-scale) to the torch data
    """

    def __init__(self, scale=0.1, clip=True):
        self.scale = scale
        self.clip = clip

    def __call__(self, tensor):
        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        noisy_tensor = (
            tensor + torch.rand(tensor.size()) * (-2 * self.scale) + self.scale
        )
        # noisy_tensor = tensor + torch.rand(tensor.size()) * (-self.scale) + self.scale
        if self.clip:
            noisy_tensor = torch.clip(noisy_tensor, 0, 1)
        return noisy_tensor

    def __repr__(self):
        return self.__class__.__name__ + "(a={0}, b={1})".format(
            -self.scale, self.scale
        )
