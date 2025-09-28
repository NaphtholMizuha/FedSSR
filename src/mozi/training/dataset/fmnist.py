from torchvision import datasets, transforms

def get_fmnist(path: str, train: bool = True):
    if train:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530]),  # FashionMNIST的均值和标准差
            transforms.RandomHorizontalFlip(),  # 可以选择保留或移除
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 可选的数据增强
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530]),  # FashionMNIST的均值和标准差
        ])
    return datasets.FashionMNIST(root=path, train=train, download=True, transform=tf)