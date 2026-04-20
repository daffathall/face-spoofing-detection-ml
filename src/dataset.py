# src/dataset.py — Upgraded dengan insight dari notebook teman

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2,
                           rotate_limit=45, border_mode=cv2.BORDER_REFLECT, p=0.7),

        # Insight dari notebook teman: ColorJitter explicit
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.2, hue=0.02, p=1.0),

        # Warna & tekstur
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50,
                                  val_shift_limit=30, p=1.0),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
        ], p=0.7),

        # Insight dari notebook teman: simulasi artefak print/screen
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            A.Downscale(scale_min=0.7, scale_max=0.9,
                        interpolation=cv2.INTER_LINEAR, p=1.0),
            A.GaussNoise(var_limit=(10, 80), p=1.0),
        ], p=0.3),

        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),

        # Insight dari notebook teman: ToGray sesekali
        A.ToGray(p=0.05),

        # Dropout
        A.CoarseDropout(max_holes=8, max_height=img_size//10,
                        max_width=img_size//10, fill_value=0, p=0.3),

        A.GridDistortion(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size):
    """
    TTA transforms — insight dari notebook teman:
    tambah center crop transform untuk lebih banyak context wajah.
    """
    norm = [A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)), ToTensorV2()]
    return [
        # 1. Original
        A.Compose([A.Resize(img_size, img_size)] + norm),
        # 2. Horizontal flip
        A.Compose([A.Resize(img_size, img_size),
                   A.HorizontalFlip(p=1.0)] + norm),
        # 3. Slight upscale + center crop (insight notebook teman)
        A.Compose([A.Resize(int(img_size * 1.15), int(img_size * 1.15)),
                   A.CenterCrop(img_size, img_size)] + norm),
        # 4. Brightness tweak
        A.Compose([A.Resize(img_size, img_size),
                   A.RandomBrightnessContrast(0.1, 0.1, p=1.0)] + norm),
        # 5. HueSat tweak
        A.Compose([A.Resize(img_size, img_size),
                   A.HueSaturationValue(10, 20, 10, p=1.0)] + norm),
    ]


class FaceDataset(Dataset):
    """Dataset untuk training dan validasi dengan label."""
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)


class TestDataset(Dataset):
    """Dataset khusus untuk data test tanpa label."""
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image


def build_samples(train_dir, class2idx):
    """Mengumpulkan path gambar dan label dari struktur folder."""
    samples = []
    for cls_name, idx in class2idx.items():
        cls_dir = os.path.join(train_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Not found: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                samples.append((os.path.join(cls_dir, fname), idx))
    return samples


def build_test_paths(test_dir):
    """Mengumpulkan path gambar test untuk inference — sorted deterministik."""
    paths = []
    for fname in sorted(os.listdir(test_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            paths.append((os.path.join(test_dir, fname),
                          os.path.splitext(fname)[0]))
    return paths


def get_weighted_sampler(samples, num_classes):
    """WeightedRandomSampler untuk handle class imbalance."""
    from torch.utils.data import WeightedRandomSampler
    counts = torch.zeros(num_classes)
    for _, lbl in samples:
        counts[lbl] += 1
    class_weights  = 1.0 / counts.clamp(min=1)
    sample_weights = torch.tensor([class_weights[lbl].item() for _, lbl in samples])
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
