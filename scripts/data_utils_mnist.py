import os
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _mnist_transform(
    normalize: bool = True,
) -> transforms.Compose:
    t = [transforms.ToTensor()]
    if normalize:
        t.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(t)


def _collate_dict(batch):
    """
    batch: list of tuples (x, y) or (x, y, idx) depending on wrapper.
    returns dict with keys: x, labels, idx
    """
    xs, ys, idxs = [], [], []
    for item in batch:
        if len(item) == 2:
            x, y = item
            idx = -1
        elif len(item) == 3:
            x, y, idx = item
        else:
            raise ValueError(f"Unexpected MNIST item length: {len(item)}")
        xs.append(x)
        ys.append(y)
        idxs.append(idx)

    x = torch.stack(xs, dim=0)
    labels = torch.tensor(ys, dtype=torch.long)
    idx = torch.tensor(idxs, dtype=torch.long)
    return {"x": x, "labels": labels, "idx": idx}


class IndexedDataset(Dataset):
    """
    Wrap any torchvision dataset so that __getitem__ returns (x, y, idx).
    """
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, idx


def build_dataloader_mnist(
    data_root: str,
    train: bool = True,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    download: bool = True,
    normalize: bool = True,
    return_dataset: bool = False,
    custom_dataset: Optional[Dataset] = None,
    limit_size: Optional[int] = None,
    drop_last: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, Dataset]]:
    """
    Build MNIST DataLoader using torchvision.datasets.MNIST.

    Args:
        data_root: where to store/download MNIST.
        train: True -> train split, False -> test split.
        batch_size: dataloader batch size.
        shuffle: shuffle dataset.
        num_workers: dataloader workers.
        pin_memory: pin_memory for CUDA.
        download: download if missing.
        normalize: whether to apply Normalize((0.1307,), (0.3081,))
        return_dataset: if True, return (loader, dataset)
        custom_dataset: if provided, use it directly (e.g., Subset for LOO).
        limit_size: if set, take only first N examples (after wrapping).
        drop_last: drop last batch.

    Returns:
        DataLoader or (DataLoader, Dataset)
    """
    os.makedirs(data_root, exist_ok=True)

    if custom_dataset is not None:
        dataset = custom_dataset
    else:
        base = datasets.MNIST(
            root=data_root,
            train=train,
            transform=_mnist_transform(normalize=normalize),
            download=download,
        )
        dataset = IndexedDataset(base)

    # Optionally truncate dataset (useful for quick experiments)
    if limit_size is not None:
        if limit_size <= 0:
            raise ValueError(f"limit_size must be >0, got {limit_size}")
        limit_size = min(limit_size, len(dataset))
        dataset = Subset(dataset, list(range(limit_size)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if custom_dataset is None else shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate_dict,
    )

    if return_dataset:
        return loader, dataset
    return loader
