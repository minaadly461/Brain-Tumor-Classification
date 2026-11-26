
from pathlib import Path

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "validation.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

CLASS_NAMES = [
    "normal",
    "glioma_tumor",
    "meningioma_tumor",
    "pituitary_tumor"
]

IMAGE_SIZE = 256
BATCH_SIZE = 16
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. Image Transformations
# ==========================================

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

    transforms.ToTensor()
])

eval_transform = transforms.ToTensor()


# ==========================================
# 3. Custom Dataset
# ==========================================

class BrainTumorDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(CLASS_NAMES)
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        label = self.class_to_idx[row["class"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# ==========================================
# 4. Load Split Manifests
# ==========================================

def load_dataframes():

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    return train_df, val_df, test_df


# ==========================================
# 5. Create DataLoaders
# ==========================================

def create_dataloaders(batch_size=BATCH_SIZE):

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
