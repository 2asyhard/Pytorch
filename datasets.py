"""
basic using method of datasets, dataloader, transformer
check out link below for additional built in pytorch dataset
https://pytorch.org/vision/stable/datasets.html

"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 32
image_size = 64 # resized image
channels_img = 1 # 1 if gray image, 3 if color image
local_data_path = 'path to local data'

def get_transform():
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), # resize image
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5), # randomly rotate image between -5 to 5 degrees
            transforms.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Normalize(
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]), # normalize image
            ])
    return transform


def get_MNIST_dataset(transform):
    data_train = datasets.MNIST(root='data/',
                                train=True,
                                transform=transform,
                                download=True)

    data_test = datasets.MNIST(root='data/',
                               train=False,
                               transform=transform,
                               download=True)

    return data_train, data_test


def get_CIFAR100_dataset(transform):
    data_train = datasets.CIFAR100(root='data/',
                                train=True,
                                transform=transform,
                                download=True)

    data_test = datasets.CIFAR100(root='data/',
                               train=False,
                               transform=transform,
                               download=True)

    return data_train, data_test


def get_fashionMNIST_dataset(transform):
    data_train = datasets.FashionMNIST(root='data/',
                                train=True,
                                transform=transform,
                                download=True)

    data_test = datasets.FashionMNIST(root='data/',
                               train=False,
                               transform=transform,
                               download=True)

    return data_train, data_test


def get_local_dataset(transform):
    # get local files as datasets
    dataset = datasets.ImageFolder(root=local_data_path, transform=transform)
    return dataset,


def get_dataloader(dataset):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers= 4, # (number of gpus * 4) is recommended
        pin_memory= True,
    )
    return loader


transform = get_transform()

# select dataset
train_data, test_data = get_MNIST_dataset(transform)
# train_data, test_data = get_CIFAR100_dataset(transform)
# train_data, test_data = get_fashionMNIST_dataset(transform)
# train_data, test_data = get_local_dataset(transform)





























