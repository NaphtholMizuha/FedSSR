import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tiny_imagenet_torch import TinyImageNet
from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    # 必须放大，否则预训练权重无法有效提取特征
    transforms.Resize(224), 
    # 数据增强
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 可选，增加难度
    transforms.ToTensor(),
    normalize,
])

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize,
])

def get_tiny_imagenet(path: str, train: bool):
    transform = transforms.ToTensor()


    return TinyImageNet(
        root=path,
        train=train,
        download=True,
        transform=train_transforms if train else val_transforms
    )
