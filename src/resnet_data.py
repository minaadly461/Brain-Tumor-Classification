
import torch
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms

from data import (
    BrainTumorDataset,
    load_dataframes,
    BATCH_SIZE
)

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transformations
train_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomRotation(10)
    ], p=0.5),

    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05)
        )
    ], p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# Validation and testing transformations
eval_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


def create_resnet_dataloaders(batch_size=BATCH_SIZE):

    train_df, val_df, test_df = load_dataframes()

    train_dataset = BrainTumorDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = BrainTumorDataset(
        val_df,
        transform=eval_transform
    )

    test_dataset = BrainTumorDataset(
        test_df,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
