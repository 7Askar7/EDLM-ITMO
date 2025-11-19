"""Production-ready data loaders with automatic dataset downloading.

This module provides automated dataset loading for IMDB, BSD300, and
MovieLens datasets. Features include:
- Automatic downloading from official sources
- Progress bars for downloads
- Dataset caching and validation
- PyTorch DataLoader creation with optimized settings

Datasets supported:
- IMDB: Sentiment analysis (Stanford)
- BSD300: Super-resolution training images (Berkeley)
- MovieLens-1M/25M: Recommendation systems (GroupLens)
"""
import csv
import gzip
import os
import shutil
import tarfile
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads with urllib.

    Extends tqdm to provide a progress bar compatible with
    urllib.request.urlretrieve's reporthook parameter.
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar based on download progress.

        Args:
            b: Number of blocks transferred so far.
            bsize: Size of each block (in bytes).
            tsize: Total size of the file (in bytes).
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar.

    Args:
        url: URL to download from.
        output_path: Local path to save the downloaded file.
    """
    print(f"Downloading {url}")
    with DownloadProgressBar(
        unit='B',
        unit_scale=True,
        miniters=1,
        desc=output_path
    ) as t:
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to
        )


# ==================== IMDB Dataset ====================

def simple_tokenizer(text):
    """Simple text tokenizer for IMDB reviews.

    Performs basic text preprocessing:
    - Converts to lowercase
    - Removes HTML tags
    - Removes non-alphanumeric characters
    - Splits on whitespace

    Args:
        text: Raw text string.

    Returns:
        List of tokens (lowercase words).
    """
    import re
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep only alphanumeric characters and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


class IMDBDataset(Dataset):
    """IMDB Dataset for binary sentiment classification.

    Attributes:
        data: List of (text, label) tuples.
        vocab: Dictionary mapping tokens to indices.
        tokenizer: Function to tokenize text.
        max_len: Maximum sequence length.
    """

    def __init__(self, data, vocab, tokenizer, max_len):
        """Initialize IMDB dataset.

        Args:
            data: List of (text, label) tuples.
            vocab: Dictionary mapping tokens to indices.
            tokenizer: Tokenization function.
            max_len: Maximum sequence length.
        """
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (indices, length, label).
        """
        text, label = self.data[idx]

        # Tokenize text
        tokens = self.tokenizer(text)

        # Convert tokens to indices (1 = <unk> for unknown words)
        indices = [self.vocab.get(token, 1) for token in tokens]

        # Truncate if too long
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]

        length = len(indices)

        # Pad to max_len
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))

        return torch.LongTensor(indices), length, label


def load_imdb_dataset(data_dir='./data/imdb', max_vocab_size=20000,
                      max_len=256):
    """Load IMDB dataset directly from Stanford with auto-download.

    Downloads the IMDB sentiment analysis dataset from Stanford's website,
    builds a vocabulary from the training set, and creates PyTorch datasets
    for training and testing.

    Args:
        data_dir: Directory to store/load the dataset (default: './data/imdb').
        max_vocab_size: Maximum vocabulary size (default: 20000).
        max_len: Maximum sequence length (default: 256).

    Returns:
        Tuple of (train_dataset, test_dataset, vocab_size) where:
            - train_dataset: IMDBDataset for training.
            - test_dataset: IMDBDataset for testing.
            - vocab_size: Size of the vocabulary.
    """
    import re
    import tarfile
    from collections import Counter

    print("Loading IMDB dataset...")

    # Create data directory
    os.makedirs(data_dir, exist_ok=True)

    # Download if not exists
    dataset_path = os.path.join(data_dir, 'aclImdb')
    if not os.path.exists(dataset_path):
        print("Downloading IMDB dataset from Stanford...")
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        tar_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')

        download_url(url, tar_path)

        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        os.remove(tar_path)
        print("Dataset extracted!")

    # Load data from files
    def load_data_from_dir(data_path):
        data = []
        for label_dir in ['pos', 'neg']:
            label = 1 if label_dir == 'pos' else 0
            dir_path = os.path.join(data_path, label_dir)
            if not os.path.exists(dir_path):
                continue
            for filename in os.listdir(dir_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(dir_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        data.append((text, label))
        return data

    print("Loading training data...")
    train_data = load_data_from_dir(os.path.join(dataset_path, 'train'))
    print(f"Loaded {len(train_data)} training samples")

    print("Loading test data...")
    test_data = load_data_from_dir(os.path.join(dataset_path, 'test'))
    print(f"Loaded {len(test_data)} test samples")

    # Build vocabulary
    print("Building vocabulary...")
    word_counts = Counter()
    for text, _ in tqdm(train_data, desc="Counting words"):
        tokens = simple_tokenizer(text)
        word_counts.update(tokens)

    # Get most common words
    most_common = word_counts.most_common(max_vocab_size - 2)  # -2 for <unk> and <pad>
    vocab = {'<pad>': 0, '<unk>': 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx

    print(f"Vocabulary size: {len(vocab)}")

    # Create PyTorch datasets
    train_dataset = IMDBDataset(train_data, vocab, simple_tokenizer, max_len)
    test_dataset = IMDBDataset(test_data, vocab, simple_tokenizer, max_len)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    return train_dataset, test_dataset, len(vocab)


# ==================== BSD Dataset for ESPCN ====================

def download_bsd300(data_dir='./data/BSD300'):
    """
    Download BSD300 dataset for super resolution
    """
    os.makedirs(data_dir, exist_ok=True)

    # BSD300 download URL
    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    tar_path = os.path.join(data_dir, 'BSDS300-images.tgz')

    if not os.path.exists(tar_path):
        download_url(url, tar_path)

        # Extract
        print("Extracting BSD300...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)

        print("BSD300 downloaded and extracted!")
    else:
        print("BSD300 already downloaded")

    # Get image paths
    image_dir = os.path.join(data_dir, 'BSDS300', 'images')

    train_dir = os.path.join(image_dir, 'train')
    test_dir = os.path.join(image_dir, 'test')

    train_images = []
    test_images = []

    if os.path.exists(train_dir):
        train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')]

    if os.path.exists(test_dir):
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]

    print(f"Found {len(train_images)} training images, {len(test_images)} test images")

    return train_images, test_images


class BSD_SR_Dataset(Dataset):
    """
    BSD dataset for super resolution
    Generates LR images on-the-fly from HR images
    """
    def __init__(self, image_paths, upscale_factor=3, patch_size=96, augment=True):
        self.image_paths = image_paths
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.augment = augment

        # Data augmentation
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load HR image
        img_path = self.image_paths[idx]
        hr_image = Image.open(img_path).convert('RGB')

        # Apply transforms
        hr_image = self.transform(hr_image)

        # Generate LR image by downsampling
        lr_size = self.patch_size // self.upscale_factor
        lr_image = transforms.Resize(
            (lr_size, lr_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        )(transforms.ToPILImage()(hr_image))
        lr_image = transforms.ToTensor()(lr_image)

        return lr_image, hr_image


def load_bsd_dataset(data_dir='./data/BSD300', upscale_factor=3, patch_size=96):
    """
    Load BSD dataset for ESPCN training

    Returns:
        train_dataset, test_dataset
    """
    print("Loading BSD300 dataset...")

    # Download if needed
    train_images, test_images = download_bsd300(data_dir)

    if len(train_images) == 0 or len(test_images) == 0:
        raise RuntimeError("BSD300 dataset not found or incomplete!")

    # Create datasets
    train_dataset = BSD_SR_Dataset(train_images, upscale_factor, patch_size, augment=True)
    test_dataset = BSD_SR_Dataset(test_images, upscale_factor, patch_size, augment=False)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    return train_dataset, test_dataset


# ==================== MovieLens-1M Dataset ====================

def download_movielens_1m(data_dir='./data/ml-1m'):
    """
    Download MovieLens-1M dataset
    """
    os.makedirs(data_dir, exist_ok=True)

    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, 'ml-1m.zip')

    if not os.path.exists(os.path.join(data_dir, 'ratings.dat')):
        if not os.path.exists(zip_path):
            download_url(url, zip_path)

        # Extract
        print("Extracting MovieLens-1M...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Move files to root
        extracted_dir = os.path.join(data_dir, 'ml-1m')
        if os.path.exists(extracted_dir):
            for file in os.listdir(extracted_dir):
                shutil.move(os.path.join(extracted_dir, file), os.path.join(data_dir, file))
            os.rmdir(extracted_dir)

        print("MovieLens-1M downloaded and extracted!")
    else:
        print("MovieLens-1M already downloaded")


class MovieLensDataset(Dataset):
    """
    MovieLens dataset for SASRec training
    """
    def __init__(self, user_sequences, num_items, max_len=200):
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.users = list(user_sequences.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        sequence = self.user_sequences[user]

        # Pad or truncate
        if len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]

        # Create input and target
        input_seq = sequence[:-1]
        target_pos = sequence[1:]

        # Pad sequences
        input_len = len(input_seq)
        if input_len < self.max_len - 1:
            input_seq = [0] * (self.max_len - 1 - input_len) + input_seq
            target_pos = [0] * (self.max_len - 1 - input_len) + target_pos

        # Add one more padding for uniformity
        input_seq.append(0)
        target_pos.append(0)

        # Sample negative items
        target_neg = []
        for _ in range(self.max_len):
            neg_item = np.random.randint(1, self.num_items + 1)
            while neg_item in sequence:
                neg_item = np.random.randint(1, self.num_items + 1)
            target_neg.append(neg_item)

        return (
            torch.LongTensor(input_seq),
            torch.LongTensor(target_pos),
            torch.LongTensor(target_neg)
        )


def download_movielens_25m(data_dir='./data/ml-25m'):
    """
    Download MovieLens-25M dataset
    """
    os.makedirs(data_dir, exist_ok=True)

    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = os.path.join(data_dir, 'ml-25m.zip')

    if not os.path.exists(os.path.join(data_dir, 'ratings.csv')):
        if not os.path.exists(zip_path):
            print("Downloading MovieLens-25M (250MB)...")
            download_url(url, zip_path)

        # Extract
        print("Extracting MovieLens-25M...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Move files to root
        extracted_dir = os.path.join(data_dir, 'ml-25m')
        if os.path.exists(extracted_dir):
            for file in os.listdir(extracted_dir):
                shutil.move(os.path.join(extracted_dir, file), os.path.join(data_dir, file))
            os.rmdir(extracted_dir)

        print("MovieLens-25M downloaded and extracted!")
    else:
        print("MovieLens-25M already downloaded")


def load_movielens_dataset(data_dir='./data/ml-1m', min_rating=4.0, max_len=200):
    """
    Load MovieLens dataset for SASRec training
    Supports both ML-1M and ML-25M

    Returns:
        train_dataset, test_dataset, num_users, num_items
    """
    # Detect which dataset based on directory name
    is_25m = 'ml-25m' in data_dir.lower()
    dataset_name = "MovieLens-25M" if is_25m else "MovieLens-1M"

    print(f"Loading {dataset_name} dataset...")

    # Download if needed
    if is_25m:
        download_movielens_25m(data_dir)
    else:
        download_movielens_1m(data_dir)

    # Read ratings - ML-25M uses CSV, ML-1M uses :: delimiter
    if is_25m:
        ratings_file = os.path.join(data_dir, 'ratings.csv')
    else:
        ratings_file = os.path.join(data_dir, 'ratings.dat')

    if not os.path.exists(ratings_file):
        raise RuntimeError(f"{os.path.basename(ratings_file)} not found in {data_dir}")

    print("Reading ratings...")
    user_sequences = defaultdict(list)
    item_set = set()

    if is_25m:
        # ML-25M uses CSV format: userId,movieId,rating,timestamp
        import csv
        with open(ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in tqdm(reader, desc="Processing ratings"):
                user_id = int(row[0])
                item_id = int(row[1])
                rating = float(row[2])
                timestamp = int(row[3])

                # Only keep high ratings
                if rating >= min_rating:
                    user_sequences[user_id].append((timestamp, item_id))
                    item_set.add(item_id)
    else:
        # ML-1M uses :: delimiter: UserID::MovieID::Rating::Timestamp
        with open(ratings_file, 'r', encoding='latin-1') as f:
            for line in tqdm(f, desc="Processing ratings"):
                parts = line.strip().split('::')
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])
                timestamp = int(parts[3])

                # Only keep high ratings
                if rating >= min_rating:
                    user_sequences[user_id].append((timestamp, item_id))
                    item_set.add(item_id)

    print(f"Found {len(user_sequences)} users, {len(item_set)} items")

    # Sort by timestamp
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[0])
        user_sequences[user_id] = [item for _, item in user_sequences[user_id]]

    # Filter users with too few interactions
    min_seq_len = 5
    user_sequences = {
        user: seq for user, seq in user_sequences.items()
        if len(seq) >= min_seq_len
    }

    print(f"After filtering: {len(user_sequences)} users")

    # Remap item IDs to be contiguous
    item_list = sorted(list(item_set))
    item_to_idx = {item: idx + 1 for idx, item in enumerate(item_list)}
    num_items = len(item_list)

    # Remap sequences
    user_sequences_remapped = {}
    for user, seq in user_sequences.items():
        user_sequences_remapped[user] = [item_to_idx[item] for item in seq]

    # Split train/test
    train_sequences = {}
    test_sequences = {}

    for user, seq in user_sequences_remapped.items():
        # Use last item for testing
        train_sequences[user] = seq[:-1]
        test_sequences[user] = seq

    # Create datasets
    train_dataset = MovieLensDataset(train_sequences, num_items, max_len)
    test_dataset = MovieLensDataset(test_sequences, num_items, max_len)

    num_users = len(user_sequences_remapped)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Users: {num_users}, Items: {num_items}")

    return train_dataset, test_dataset, num_users, num_items


# ==================== Helper Functions ====================

def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True,
                   prefetch_factor=2, persistent_workers=None):
    """
    Create DataLoader with standard settings
    """
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    if num_workers > 0:
        if persistent_workers is None:
            persistent_workers = True
        dataloader_kwargs['persistent_workers'] = persistent_workers
        dataloader_kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(**dataloader_kwargs)
