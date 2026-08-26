"""
Data Loader for QuantumShieldFL

Handles downloading MNIST and partitioning it across federated clients.
Supports both IID and non-IID (Dirichlet-based) data distributions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def get_mnist_transforms():
    """Standard MNIST normalization transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])


def load_mnist(data_dir=None):
    """
    Download and load MNIST train and test datasets.
    
    Args:
        data_dir: Directory to store downloaded data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    transform = get_mnist_transforms()
    
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset


def partition_iid(dataset, num_clients):
    """
    Partition dataset into equal IID shards for each client.
    
    Each client gets a random, equal-sized subset of the data with
    similar class distributions.
    
    Args:
        dataset: PyTorch dataset to partition
        num_clients: Number of FL clients
        
    Returns:
        dict mapping client_id → list of dataset indices
    """
    np.random.seed(config.RANDOM_SEED)
    num_items = len(dataset)
    all_indices = np.arange(num_items)
    np.random.shuffle(all_indices)
    
    # Split into equal chunks
    chunks = np.array_split(all_indices, num_clients)
    
    client_data = {}
    for client_id in range(num_clients):
        client_data[client_id] = chunks[client_id].tolist()
    
    return client_data


def partition_non_iid(dataset, num_clients, alpha=None):
    """
    Partition dataset using a Dirichlet distribution for non-IID splits.
    
    Lower alpha → more heterogeneous (each client gets fewer classes).
    Higher alpha → more homogeneous (approaches IID).
    
    Args:
        dataset: PyTorch dataset to partition
        num_clients: Number of FL clients
        alpha: Dirichlet concentration parameter
        
    Returns:
        dict mapping client_id → list of dataset indices
    """
    if alpha is None:
        alpha = config.NON_IID_ALPHA
    
    np.random.seed(config.RANDOM_SEED)
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    client_data = {i: [] for i in range(num_clients)}
    
    # For each class, distribute indices according to Dirichlet
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Convert proportions to actual counts
        proportions = (proportions * len(indices)).astype(int)
        
        # Fix rounding errors: assign remaining to first client
        diff = len(indices) - proportions.sum()
        proportions[0] += diff
        
        # Assign indices to clients
        start = 0
        for client_id in range(num_clients):
            end = start + proportions[client_id]
            client_data[client_id].extend(indices[start:end].tolist())
            start = end
    
    return client_data


def get_client_dataloaders(num_clients=None, batch_size=None, non_iid=None):
    """
    Create DataLoaders for each federated learning client.
    
    Args:
        num_clients: Number of FL clients
        batch_size: Training batch size
        non_iid: Whether to use non-IID partitioning
        
    Returns:
        Tuple of (client_loaders: dict, test_loader: DataLoader)
        where client_loaders maps client_id → DataLoader
    """
    if num_clients is None:
        num_clients = config.NUM_CLIENTS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if non_iid is None:
        non_iid = config.NON_IID
    
    train_dataset, test_dataset = load_mnist()
    
    # Partition training data
    if non_iid:
        client_indices = partition_non_iid(train_dataset, num_clients)
    else:
        client_indices = partition_iid(train_dataset, num_clients)
    
    # Create DataLoaders for each client
    client_loaders = {}
    for client_id in range(num_clients):
        subset = Subset(train_dataset, client_indices[client_id])
        client_loaders[client_id] = DataLoader(
            subset, batch_size=batch_size, shuffle=True, drop_last=False
        )
    
    # Global test loader for evaluation
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    
    # Print distribution summary
    print(f"\n{'='*50}")
    print(f"Data Distribution Summary ({'Non-IID' if non_iid else 'IID'})")
    print(f"{'='*50}")
    for cid in range(num_clients):
        n_samples = len(client_indices[cid])
        print(f"  Client {cid}: {n_samples} samples")
    print(f"  Test set: {len(test_dataset)} samples")
    print(f"{'='*50}\n")
    
    return client_loaders, test_loader
