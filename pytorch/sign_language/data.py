import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from . import config


def build_transforms():
    return transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(list(config.IMAGENET_MEAN), list(config.IMAGENET_STD)),
        ]
    )


def build_eval_transforms():
    """No augmentation — used for validation, test, and camera inference."""
    return transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(list(config.IMAGENET_MEAN), list(config.IMAGENET_STD)),
        ]
    )


def get_class_names(data_root=None):
    """Class names according to ImageFolder order in the training directory (for prediction labels)."""
    if data_root is None:
        data_root = config.DEFAULT_DATA_ROOT
    train_path = os.path.join(data_root, "train")
    ds = datasets.ImageFolder(root=train_path, transform=transforms.ToTensor())
    return ds.classes


def load_datasets(base_path=None, data_transforms=None):
    if base_path is None:
        base_path = config.DEFAULT_DATA_ROOT
    if data_transforms is None:
        data_transforms = build_transforms()

    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    full_train_data = datasets.ImageFolder(root=train_path, transform=data_transforms)
    eval_transforms = build_eval_transforms()
    test_data = datasets.ImageFolder(root=test_path, transform=eval_transforms)

    train_size = int(config.TRAIN_VAL_SPLIT * len(full_train_data))
    val_size = len(full_train_data) - train_size

    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    return train_data, val_data, test_data, full_train_data


def make_dataloaders(train_data, val_data, test_data, batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
