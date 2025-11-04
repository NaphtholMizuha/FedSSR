from torch.utils.data import Dataset
from .cifar import get_cifar10, get_cifar100
from .imagenette import get_imagenette, get_imagewoof
from .fmnist import get_fmnist
from .food101 import get_food101
from .tiny_imagenet import get_tiny_imagenet

def fetch_dataset(path: str, dataset: str) -> tuple[Dataset, Dataset]:
    if dataset == 'cifar10':
        return get_cifar10(path, True), get_cifar10(path, False)
    elif dataset == 'cifar100':
        return get_cifar100(path, True), get_cifar100(path, False)
    elif dataset == 'imagenette':
        return get_imagenette(path, True), get_imagenette(path, False)
    elif dataset == 'imagewoof':
        return get_imagewoof(path, True), get_imagewoof(path, False)
    elif dataset == 'fmnist':
        return get_fmnist(path, True), get_fmnist(path, False)
    elif dataset == 'food101':
        return get_food101(path, True), get_food101(path, False)
    elif dataset == 'tiny-imagenet':
        return get_tiny_imagenet(path, True), get_tiny_imagenet(path, False)
    else:
        raise Exception(f"Unsupported Dataset: {dataset}")