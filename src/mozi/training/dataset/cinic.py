from pytorch_cinic.dataset import CINIC10

from torchvision import datasets, transforms
from torch.utils.data import Dataset


def get_cinic10(path: str, train: bool = True):
    if train:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
    return CINIC10(root=path, partition="train" if train else "test", download=True, transform=tf)


