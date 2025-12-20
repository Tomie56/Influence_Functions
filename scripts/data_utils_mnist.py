import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

# MNIST default transform
_DEFAULT_MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def mnist_collate_fn(batch):
    """Collate function for MNIST dataset"""
    xs = torch.stack([item[0] for item in batch])
    ys = torch.tensor([item[1] for item in batch])
    ids = torch.tensor([item[2] if len(item) == 3 else idx for idx, item in enumerate(batch)])
    return xs, ys, ids

def build_dataloader_mnist(
    data_root: str,
    train: bool = True,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    limit_size: int = -1,
    transform=None,
    return_dataset: bool = False
):
    """Build MNIST dataloader/dataset with optional size limit"""
    if transform is None:
        transform = _DEFAULT_MNIST_TRANSFORM
    
    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(
        root=data_root,
        train=train,
        download=True,
        transform=transform
    )

    # Add sample IDs
    class MNISTWithID(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            x, y = self.base_dataset[idx]
            return x, y, idx
    
    dataset_with_id = MNISTWithID(mnist_dataset)

    # Limit dataset size if needed
    if limit_size > 0 and limit_size < len(dataset_with_id):
        dataset_with_id = Subset(dataset_with_id, list(range(limit_size)))

    # Build dataloader
    dataloader = DataLoader(
        dataset_with_id,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mnist_collate_fn
    )
    
    if return_dataset:
        return dataloader, dataset_with_id
    
    return dataloader, None