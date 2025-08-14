import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class IAMDataset(Dataset):
    """IAM Handwriting Dataset loader and preprocessor"""

    def __init__(self, root_dir, mode='train', img_height=32, img_width=128, mean=0.5, std=0.5):
        self.root_dir = root_dir
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean
        self.std = std

        # Load data paths and labels
        self.samples = self._load_samples()

        # Transformations
        base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])

        # Augmentations for training
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_height, img_width)),
                transforms.RandomAffine(
                    degrees=10,  # Increased from ±5 to ±10°
                    shear=10,  # More extreme italicization
                    scale=(0.8, 1.2)  # Wider zoom range
                ),
                transforms.ColorJitter(
                    brightness=0.3,  # Brighter/darker variations
                    contrast=0.3,
                    saturation=0.2
                ),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),  # Blur
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean], std=[std]),
                AddGaussianNoise(mean=0., std=0.05)  # Custom noise
            ])
        else:
            self.transform = base_transform

    def _load_samples(self):
        # Implement IAM dataset parsing
        # This should return a list of (image_path, label) tuples
        # For simplicity, we'll show a placeholder implementation

        samples = []
        words_file = os.path.join(self.root_dir, "words.txt")

        with open(words_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 9:
                    # Format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A
                    file_parts = parts[0].split('-')
                    dir1 = file_parts[0]
                    dir2 = f"{file_parts[0]}-{file_parts[1]}"
                    file_name = f"{parts[0]}.png"
                    img_path = os.path.join(self.root_dir, "words", dir1, dir2, file_name)

                    # The label is the 8th element onward joined by space
                    label = ' '.join(parts[8:])

                    if os.path.exists(img_path):
                        samples.append((img_path, label))

        # Split into train/val/test (80/10/10)
        np.random.seed(42)
        np.random.shuffle(samples)

        if self.mode == 'train':
            return samples[:int(0.8 * len(samples))]
        elif self.mode == 'val':
            return samples[int(0.8 * len(samples)):int(0.9 * len(samples))]
        else:  # test
            return samples[int(0.9 * len(samples)):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image with validation
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error loading: {img_path}")  # Debug which file failed
            # Option 1: Skip by returning another random sample
            return self[np.random.randint(0, len(self))]

        # Apply transformations
        img = self.transform(img)

        # Convert label to tensor
        target = torch.tensor([self._char_to_index(c) for c in label], dtype=torch.long)

        return img, target, label

    def _char_to_index(self, char):
        """Convert character to index in vocabulary"""
        if not hasattr(config, 'VOCAB'):
            raise ValueError("config.VOCAB is not defined. Please check your config.py")
        return config.VOCAB.find(char) if char in config.VOCAB else len(config.VOCAB)


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    images = []
    targets = []
    labels = []
    max_width = 0

    for img, target, label in batch:
        images.append(img)
        targets.append(target)
        labels.append(label)
        if img.shape[2] > max_width:
            max_width = img.shape[2]

    # Pad images to max width in batch
    padded_images = torch.zeros(len(images), 1, config.IMG_HEIGHT, max_width)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.shape[2]] = img

    # Pad targets with -1 (ignore index)
    max_target_len = max(len(t) for t in targets)
    padded_targets = torch.ones(len(targets), max_target_len, dtype=torch.long) * -1
    for i, target in enumerate(targets):
        padded_targets[i, :len(target)] = target

    return padded_images, padded_targets, labels