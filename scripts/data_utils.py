import torch
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset

class MultimodalJsonlDataset(Dataset):
    """Multimodal dataset loaded from jsonl file (text + optional image)"""
    def __init__(
        self,
        jsonl_path: str,
        image_root: str = "./",
        max_seq_len: int = 768
    ):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.max_seq_len = max_seq_len
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load samples from jsonl file"""
        samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                # Ensure required fields exist
                assert "input_ids" in sample, "Sample missing 'input_ids'"
                assert "labels" in sample, "Sample missing 'labels'"
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        sample_id = sample.get("id", idx)
        
        # Truncate/pad input_ids to max_seq_len
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        elif len(input_ids) < self.max_seq_len:
            input_ids = torch.cat([
                input_ids,
                torch.zeros(self.max_seq_len - len(input_ids), dtype=torch.long)
            ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "idx": sample_id
        }

def build_dataloader(
    data_path: str,
    tokenizer=None,
    image_root: str = "./",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    limit_size: int = -1,
    max_seq_len: int = 768,
    return_dataset: bool = False
):
    """Build dataloader for multimodal jsonl dataset"""
    # Build dataset
    dataset = MultimodalJsonlDataset(
        jsonl_path=data_path,
        image_root=image_root,
        max_seq_len=max_seq_len
    )

    # Limit dataset size if needed
    if limit_size > 0 and limit_size < len(dataset):
        dataset = Subset(dataset, list(range(limit_size)))

    if return_dataset:
        return None, dataset

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "labels": torch.stack([item["labels"] for item in x]),
            "idx": torch.tensor([item["idx"] for item in x])
        }
    )
    return dataloader, None