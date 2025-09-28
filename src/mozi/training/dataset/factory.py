from torch.utils.data import Dataset
from .cifar import get_cifar10
from .imagenette import get_imagenette, get_imagewoof
from .fmnist import get_fmnist
from .food101 import get_food101

def fetch_dataset(path: str, dataset: str) -> tuple[Dataset, Dataset]:
    if dataset == 'cifar10':
        return get_cifar10(path, True), get_cifar10(path, False)
    elif dataset == 'imagenette':
        return get_imagenette(path, True), get_imagenette(path, False)
    elif dataset == 'imagewoof':
        return get_imagewoof(path, True), get_imagewoof(path, False)
    elif dataset == 'fmnist':
        return get_fmnist(path, True), get_fmnist(path, False)
    elif dataset == 'food101':
        return get_food101(path, True), get_food101(path, False)
    else:
        raise Exception(f"Unsupported Dataset: {dataset}")