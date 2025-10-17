from torchvision import datasets, transforms
from torch.utils.data import Dataset


def get_cifar10(path: str, train: bool = True):
    if train:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomCrop(32, padding=4),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
    return datasets.CIFAR10(root=path, train=train, download=True, transform=tf)


class _CIFAR100Wrapper(Dataset):
    def __init__(self, *args, **kwargs):
        self.cifar100 = datasets.CIFAR100(*args, **kwargs)

    def __getitem__(self, index):
        img, label = self.cifar100[index]
        return img, label

    def __len__(self):
        return len(self.cifar100)


def get_cifar100(path: str, train: bool = True):
    if train:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomCrop(32, padding=4),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761]),
        ])
    return _CIFAR100Wrapper(root=path, train=train, download=True, transform=tf)