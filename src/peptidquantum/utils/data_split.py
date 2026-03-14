"""Data splitting utilities for train/val/test"""
from __future__ import annotations

import random
from typing import TypeVar, List

T = TypeVar('T')


def split_data(
    data: List[T],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple[List[T], List[T], List[T]]:
    """
    Split data into train/val/test sets
    
    Args:
        data: List of data items
        train_ratio: Ratio for training set (default 0.8)
        val_ratio: Ratio for validation set (default 0.1)
        test_ratio: Ratio for test set (default 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    n = len(data_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]
    
    return train_data, val_data, test_data


def stratified_split(
    data: List[T],
    labels: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple[List[T], List[T], List[T]]:
    """
    Stratified split maintaining class balance
    
    Args:
        data: List of data items
        labels: List of labels (0 or 1)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert len(data) == len(labels), "Data and labels must have same length"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    
    # Separate by class
    pos_data = [d for d, l in zip(data, labels) if l == 1]
    neg_data = [d for d, l in zip(data, labels) if l == 0]
    
    # Split each class
    pos_train, pos_val, pos_test = split_data(pos_data, train_ratio, val_ratio, test_ratio, seed)
    neg_train, neg_val, neg_test = split_data(neg_data, train_ratio, val_ratio, test_ratio, seed)
    
    # Combine and shuffle
    train_data = pos_train + neg_train
    val_data = pos_val + neg_val
    test_data = pos_test + neg_test
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data
