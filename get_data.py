import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import random
import numpy as np


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class BaseSimCLRException(Exception):
    """Base exception"""

class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar100': lambda: datasets.CIFAR100(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(64),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


def get_stl10_unlabeled_vanilla_deform(batch_size, size):
    transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.ToTensor(),
    ])
    train = datasets.STL10('./data', split='unlabeled', transform=transform, download=True)
    if size != len(train):
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), size))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader, None


def get_stl10_unlabeled_deform(batch_size, size):
    """This returns a list of 2 views of a dataset"""
    dset = ContrastiveLearningDataset('./data')
    train = dset.get_dataset('stl10', 2)
    if size != len(train):
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), size))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,)
    return train_loader, None


def get_stl10_unlabeled_patches(batch_size, size):
    transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4120], std=[0.2570]),
    ])
    train = datasets.STL10('./data', split='unlabeled', transform=transform, download=True)
    if size != len(train):
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), size))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader, None


def get_stl10_labeled(batch_size, pars):
    compose_train = []
    if pars.augment_stl_train:
        compose_train.extend([transforms.RandomCrop(64), 
                              transforms.RandomHorizontalFlip()])
    else:
        compose_train.append(transforms.CenterCrop(64))

    if pars.gaze_shift:
        compose_train.extend([transforms.Grayscale(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.4120], std=[0.2570])])
    else:
        compose_train.append(transforms.ToTensor())

    compose_test = [transforms.CenterCrop(64)]
    if pars.gaze_shift:
        compose_test.extend([transforms.Grayscale(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.4120], std=[0.2570])])
    else:
        compose_test.append(transforms.ToTensor())

    transform_train = transforms.Compose(compose_train)

    transform_test = transforms.Compose(compose_test)

    train = datasets.STL10('./data', split='train', transform=transform_train, download=True)
    test = datasets.STL10('./data', split='test', transform=transform_test, download=True)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_mnist(batch_size, size=60000):
    train = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    test = datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    
    if size != len(train):
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), size))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_cifar100(batch_size, size=60000):
    train = datasets.CIFAR100('./data', train=True, transform=transforms.ToTensor(), download=True)
    test = datasets.CIFAR100('./data', train=False, transform=transforms.ToTensor(), download=True)
    
    if size != len(train):
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), size))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_cifar10(batch_size, size=60000):
    train = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
    test = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
    
    if size != len(train):
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), size))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader



# def get_dataset(data, batch_size, size):
#     if data == 'mnist':
#         return get_mnist(batch_size, size)
#     elif data == 'Cifar100':
#         return get_cifar100(batch_size, size)
#     elif data == 'Cifar10':
#         return get_cifar10(batch_size, size)
#     elif data == 'STL10_unlabeled':
#         return get_stl10_unlabeled(batch_size, size)
#     elif data == 'STL10_labeled':
#         return get_stl10_labeled(batch_size)