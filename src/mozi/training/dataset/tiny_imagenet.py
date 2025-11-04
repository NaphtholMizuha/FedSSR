import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tiny_imagenet_torch import TinyImageNet

def get_tiny_imagenet(path: str, train: bool):
    transform = transforms.ToTensor()

    return TinyImageNet(
        root=path,
        train=train,
        download=True,
        transform=transform
    )
