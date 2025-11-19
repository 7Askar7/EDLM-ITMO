"""Dataset utilities for LSTM, ESPCN, and SASRec models.

This module provides PyTorch Dataset classes and data loading utilities for:
- LSTM text classification (IMDB sentiment analysis)
- ESPCN super-resolution (image upscaling)
- SASRec sequential recommendation (MovieLens)

The module also includes synthetic dataset generators for testing.
"""
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# ==================== LSTM Text Classification Dataset ====================

class IMDBDataset(Dataset):
    """IMDB dataset for sentiment analysis.

    This is a simple wrapper that uses torchtext datasets for binary
    sentiment classification (positive/negative reviews).

    Attributes:
        data: List of (text, label) tuples.
        vocab: Vocabulary object mapping tokens to indices.
        max_len: Maximum sequence length for truncation/padding.
        tokenizer: Basic English tokenizer from torchtext.
    """

    def __init__(self, data, vocab, max_len=256):
        """Initialize IMDB dataset.

        Args:
            data: List of (text, label) tuples where label is 0 or 1.
            vocab: Vocabulary object from torchtext.
            max_len: Maximum sequence length (default: 256).
        """
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = get_tokenizer('basic_english')

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (indices, length, label) where:
                - indices: LongTensor of token indices (padded to max_len).
                - length: Actual sequence length before padding.
                - label: Binary sentiment label (0 or 1).
        """
        text, label = self.data[idx]

        # Tokenize text into words
        tokens = self.tokenizer(text)

        # Convert tokens to vocabulary indices
        indices = [self.vocab[token] for token in tokens]

        # Truncate if sequence is too long
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]

        length = len(indices)

        # Pad sequence to max_len with zeros
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))

        return torch.LongTensor(indices), length, label


def load_imdb_data(max_vocab_size=20000, max_len=256):
    """Load IMDB dataset using torchtext.

    Downloads and loads the IMDB sentiment analysis dataset, builds a
    vocabulary from the training data, and creates PyTorch datasets.

    Args:
        max_vocab_size: Maximum vocabulary size (default: 20000).
        max_len: Maximum sequence length (default: 256).

    Returns:
        Tuple of (train_dataset, test_dataset, vocab) where:
            - train_dataset: IMDBDataset for training.
            - test_dataset: IMDBDataset for testing.
            - vocab: Vocabulary object from torchtext.

    Raises:
        ImportError: If torchtext is not installed (auto-installs).
    """
    try:
        from torchtext.datasets import IMDB
    except ImportError:
        print("torchtext not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'torchtext'])
        from torchtext.datasets import IMDB

    # Load data
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')

    # Build vocabulary
    tokenizer = get_tokenizer('basic_english')

    def yield_tokens(data_iter):
        for label, text in data_iter:
            yield tokenizer(text)

    # Rebuild iterator for vocab building
    train_iter_vocab = IMDB(split='train')
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter_vocab),
        specials=['<unk>', '<pad>'],
        max_tokens=max_vocab_size
    )
    vocab.set_default_index(vocab['<unk>'])

    # Convert to list for dataset (torchtext IMDB uses labels 1 and 2)
    train_data = [
        (text, 1 if label == 2 else 0)
        for label, text in IMDB(split='train')
    ]
    test_data = [
        (text, 1 if label == 2 else 0)
        for label, text in IMDB(split='test')
    ]

    train_dataset = IMDBDataset(train_data, vocab, max_len)
    test_dataset = IMDBDataset(test_data, vocab, max_len)

    return train_dataset, test_dataset, vocab


# ==================== ESPCN Super Resolution Dataset ====================

class SRDataset(Dataset):
    """Super Resolution Dataset for ESPCN training.

    Generates low-resolution images from high-resolution images on-the-fly
    by downsampling. Supports data augmentation including random crops and
    horizontal flips.

    Attributes:
        image_paths: List of paths to high-resolution images.
        upscale_factor: Upscaling factor (e.g., 2, 3, 4).
        patch_size: Size of HR patches to extract.
        augment: Whether to apply data augmentation.
        hr_transform: Transformation pipeline for HR images.
    """

    def __init__(self, image_paths, upscale_factor=3, patch_size=64,
                 augment=True):
        """Initialize Super Resolution dataset.

        Args:
            image_paths: List of paths to high-resolution images.
            upscale_factor: Upscaling factor (default: 3).
            patch_size: Size of patches to extract from HR images (default: 64).
            augment: Whether to apply data augmentation (default: True).
        """
        self.image_paths = image_paths
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.augment = augment

        # Transforms for HR images: crop and convert to tensor
        crop_transform = (transforms.RandomCrop(patch_size) if augment
                          else transforms.CenterCrop(patch_size))
        self.hr_transform = transforms.Compose([
            crop_transform,
            transforms.ToTensor()
        ])

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a single LR-HR image pair.

        Args:
            idx: Index of the image.

        Returns:
            Tuple of (lr_image, hr_image) where both are tensors of shape
            (3, height, width) with values in range [0, 1].
        """
        # Load high-resolution image
        img_path = self.image_paths[idx]
        hr_image = Image.open(img_path).convert('RGB')

        # Apply random horizontal flip for data augmentation
        if self.augment and random.random() > 0.5:
            hr_image = hr_image.transpose(Image.FLIP_LEFT_RIGHT)

        # Extract HR patch and convert to tensor
        hr_image = self.hr_transform(hr_image)

        # Generate LR image by bicubic downsampling
        lr_size = self.patch_size // self.upscale_factor
        lr_image = transforms.Resize(
            (lr_size, lr_size),
            interpolation=Image.BICUBIC
        )(transforms.ToPILImage()(hr_image))
        lr_image = transforms.ToTensor()(lr_image)

        return lr_image, hr_image


def create_sr_dataset_from_folder(folder_path, upscale_factor=3,
                                  patch_size=64):
    """Create super resolution dataset from a folder of images.

    Recursively searches for image files in the specified folder and creates
    a SRDataset for training ESPCN models.

    Args:
        folder_path: Path to folder containing images.
        upscale_factor: Upscaling factor (default: 3).
        patch_size: Patch size for HR images (default: 64).

    Returns:
        SRDataset initialized with all found images.
    """
    # Collect all valid image files from folder
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_paths.append(os.path.join(root, file))

    print(f"Found {len(image_paths)} images in {folder_path}")

    return SRDataset(image_paths, upscale_factor, patch_size)


def download_bsd_dataset(data_dir='./data/BSD'):
    """Download BSD100 dataset for super resolution.

    This is a placeholder function - downloads must be done manually.

    Args:
        data_dir: Directory to store the BSD dataset (default: './data/BSD').

    Note:
        The function only creates the directory and prints download
        instructions. Actual dataset must be downloaded manually from
        the Berkeley website.
    """
    print("Please download BSD100/BSD300 dataset manually from:")
    print("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/")
    print(f"And place images in {data_dir}")
    os.makedirs(data_dir, exist_ok=True)


# ==================== SASRec Recommendation Dataset ====================

class SASRecDataset(Dataset):
    """Dataset for SASRec sequential recommendation.

    Implements the training protocol for Self-Attentive Sequential
    Recommendation (SASRec) with negative sampling for BPR loss.

    Attributes:
        user_sequences: Dictionary mapping user_id to list of item_ids.
        num_items: Total number of unique items in the catalog.
        max_len: Maximum sequence length for padding/truncation.
        num_negatives: Number of negative samples per positive.
        users: List of user IDs for iteration.
    """

    def __init__(self, user_sequences, num_items, max_len=200,
                 num_negatives=1):
        """Initialize SASRec dataset.

        Args:
            user_sequences: Dictionary mapping user_id to list of item_ids.
            num_items: Total number of items in catalog.
            max_len: Maximum sequence length (default: 200).
            num_negatives: Number of negative samples per position (default: 1).
        """
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.num_negatives = num_negatives

        # Convert dictionary keys to list for indexing
        self.users = list(user_sequences.keys())

    def __len__(self):
        """Return the number of users in the dataset."""
        return len(self.users)

    def __getitem__(self, idx):
        """Get a single training sample for a user.

        Args:
            idx: Index of the user.

        Returns:
            Tuple of (input_seq, target_pos, target_neg) where:
                - input_seq: Input sequence (LongTensor).
                - target_pos: Positive target items (LongTensor).
                - target_neg: Negative sampled items (LongTensor).
        """
        user = self.users[idx]
        sequence = self.user_sequences[user]

        # Truncate or pad sequence to max_len
        if len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]  # Keep most recent items
        else:
            # Left-pad with zeros (padding token = 0)
            sequence = [0] * (self.max_len - len(sequence)) + sequence

        # Create input and target sequences (shifted by one position)
        input_seq = sequence[:-1] + [0]  # Input: all but last, plus padding
        target_pos = sequence[1:] + [0]  # Target: shifted by one position

        # Sample negative items for each position
        target_neg = []
        for _ in range(len(input_seq)):
            # Sample item not in user's history
            neg_item = random.randint(1, self.num_items)
            while neg_item in sequence:
                neg_item = random.randint(1, self.num_items)
            target_neg.append(neg_item)

        return (
            torch.LongTensor(input_seq),
            torch.LongTensor(target_pos),
            torch.LongTensor(target_neg)
        )


def load_movielens_data(data_dir='./data/ml-1m', min_rating=4.0):
    """Load MovieLens-1M dataset for sequential recommendation.

    Processes the MovieLens-1M dataset, filters by rating threshold, and
    creates train/test splits for SASRec training.

    Args:
        data_dir: Directory containing MovieLens data files (default:
            './data/ml-1m').
        min_rating: Minimum rating to consider as positive interaction
            (default: 4.0).

    Returns:
        Tuple of (train_dataset, test_dataset, num_users, num_items) where:
            - train_dataset: SASRecDataset for training.
            - test_dataset: SASRecDataset for testing.
            - num_users: Number of users after filtering.
            - num_items: Number of unique items.

    Note:
        Returns (None, None, 0, 0) if dataset files are not found.
    """
    ratings_file = os.path.join(data_dir, 'ratings.dat')

    if not os.path.exists(ratings_file):
        print(f"MovieLens-1M not found at {data_dir}")
        print("Please download from: https://grouplens.org/datasets/movielens/1m/")
        print("And extract to", data_dir)
        return None, None, 0, 0

    # Read ratings
    user_sequences = defaultdict(list)
    item_set = set()

    with open(ratings_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            user_id = int(parts[0])
            item_id = int(parts[1])
            rating = float(parts[2])
            timestamp = int(parts[3])

            # Only keep high ratings
            if rating >= min_rating:
                user_sequences[user_id].append((timestamp, item_id))
                item_set.add(item_id)

    # Sort by timestamp
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[0])
        user_sequences[user_id] = [item for _, item in user_sequences[user_id]]

    # Remove users with too few interactions
    min_seq_len = 5
    user_sequences = {
        user: seq for user, seq in user_sequences.items()
        if len(seq) >= min_seq_len
    }

    # Remap item IDs to be contiguous starting from 1
    item_list = sorted(list(item_set))
    item_to_idx = {item: idx + 1 for idx, item in enumerate(item_list)}  # Start from 1
    num_items = len(item_list)

    # Remap sequences
    user_sequences_remapped = {}
    for user, seq in user_sequences.items():
        user_sequences_remapped[user] = [item_to_idx[item] for item in seq]

    # Split into train and test
    train_sequences = {}
    test_sequences = {}

    for user, seq in user_sequences_remapped.items():
        # Use last item for testing
        train_sequences[user] = seq[:-1]
        test_sequences[user] = seq

    # Create datasets
    train_dataset = SASRecDataset(train_sequences, num_items, max_len=200)
    test_dataset = SASRecDataset(test_sequences, num_items, max_len=200)

    num_users = len(user_sequences_remapped)

    return train_dataset, test_dataset, num_users, num_items


# ==================== Synthetic Dataset Generators ====================

def create_synthetic_text_dataset(num_samples=1000, vocab_size=5000, max_len=128):
    """
    Create synthetic text classification dataset for testing

    Args:
        num_samples: Number of samples
        vocab_size: Vocabulary size
        max_len: Maximum sequence length

    Returns:
        train_dataset, test_dataset, vocab_size
    """
    # Generate random sequences
    train_data = []
    for _ in range(num_samples):
        seq_len = random.randint(10, max_len)
        indices = torch.randint(1, vocab_size, (seq_len,))
        # Label based on sum (synthetic pattern)
        label = 1 if indices.sum() % 2 == 0 else 0
        train_data.append((indices, seq_len, label))

    test_data = []
    for _ in range(num_samples // 5):
        seq_len = random.randint(10, max_len)
        indices = torch.randint(1, vocab_size, (seq_len,))
        label = 1 if indices.sum() % 2 == 0 else 0
        test_data.append((indices, seq_len, label))

    return train_data, test_data, vocab_size


def create_synthetic_sr_dataset(num_samples=100, upscale_factor=3, patch_size=64):
    """
    Create synthetic super resolution dataset for testing

    Args:
        num_samples: Number of samples
        upscale_factor: Upscaling factor
        patch_size: Patch size

    Returns:
        Dataset with synthetic images
    """
    class SyntheticSRDataset(Dataset):
        def __init__(self, num_samples, upscale_factor, patch_size):
            self.num_samples = num_samples
            self.upscale_factor = upscale_factor
            self.patch_size = patch_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random HR image
            hr_image = torch.rand(3, self.patch_size, self.patch_size)

            # Generate LR image by downsampling
            lr_size = self.patch_size // self.upscale_factor
            lr_image = torch.nn.functional.interpolate(
                hr_image.unsqueeze(0),
                size=(lr_size, lr_size),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)

            return lr_image, hr_image

    return SyntheticSRDataset(num_samples, upscale_factor, patch_size)


def create_synthetic_sasrec_dataset(num_users=1000, num_items=500, avg_seq_len=50, max_len=200):
    """
    Create synthetic SASRec dataset for testing

    Args:
        num_users: Number of users
        num_items: Number of items
        avg_seq_len: Average sequence length
        max_len: Maximum sequence length

    Returns:
        train_dataset, test_dataset, num_users, num_items
    """
    user_sequences = {}

    for user_id in range(1, num_users + 1):
        seq_len = random.randint(10, avg_seq_len)
        sequence = [random.randint(1, num_items) for _ in range(seq_len)]
        user_sequences[user_id] = sequence

    # Split train/test
    train_sequences = {user: seq[:-1] for user, seq in user_sequences.items() if len(seq) > 1}
    test_sequences = {user: seq for user, seq in user_sequences.items() if len(seq) > 1}

    train_dataset = SASRecDataset(train_sequences, num_items, max_len)
    test_dataset = SASRecDataset(test_sequences, num_items, max_len)

    return train_dataset, test_dataset, num_users, num_items
