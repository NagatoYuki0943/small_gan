import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def get_mnist(data_dir: str, batch_size: int = 128):
    transform = transforms.Compose([
        transforms.Pad(2),  # 28 -> 32
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    train_dataset = datasets.MNIST(data_dir, download=True, train=True, transform=transform)
    valid_dataset = datasets.MNIST(data_dir, download=True, train=False, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"train dataset number: {len(train_dataset)}, valid dataset number: {len(valid_dataset)}")
    print(f"train dataloader number: {len(train_dataloader)}, valid dataloader number: {len(valid_dataloader)}")

    return train_dataloader, valid_dataloader


def get_cifar(data_dir: str, batch_size: int = 128, variance: str = "CIFAR10"):
    dataset = datasets.CIFAR10 if variance == "CIFAR10" else datasets.CIFAR100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = dataset(data_dir, download=True, train=True, transform=transform)
    valid_dataset = dataset(data_dir, download=True, train=False, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"train dataset number: {len(train_dataset)}, valid dataset number: {len(valid_dataset)}")
    print(f"train dataloader number: {len(train_dataloader)}, valid dataloader number: {len(valid_dataloader)}")

    return train_dataloader, valid_dataloader


def test():
    data_dir = "datasets"
    train_dataloader, valid_dataloader = get_mnist(data_dir)
    images, ids = next(iter(train_dataloader))
    print(images.shape)
    print(ids)


if __name__ == '__main__':
    test()

