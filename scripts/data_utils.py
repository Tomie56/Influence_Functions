import torch
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms

class MultimodalJsonlDataset(Dataset):
    """Multimodal dataset (text + image) for influence function experiments, loaded from jsonl"""
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        image_root: str = "./",
        max_seq_len: int = 768
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) 
        self.max_seq_len = max_seq_len
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load and preprocess raw jsonl samples"""
        samples = []
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Jsonl file not found: {self.jsonl_path}")
        
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_idx}, error: {e}, skip this line")
                    continue
                
                assert "conversations" in sample, f"Sample at line {line_idx} missing 'conversations'"
                assert "image" in sample, f"Sample at line {line_idx} missing 'image'"
                
                # Extract human and gpt text from conversations
                human_text, gpt_text = "", ""
                for conv in sample["conversations"]:
                    if conv["from"] == "human":
                        human_text = conv["value"].replace("<image>", "").strip()
                    elif conv["from"] == "gpt":
                        gpt_text = conv["value"].strip()
                
                if not human_text:
                    print(f"Warning: Empty human text at line {line_idx}, skip this sample")
                    continue
                
                # Encode input text to input_ids
                encoded_input = self.tokenizer(
                    human_text,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = encoded_input["input_ids"].squeeze(0)
                
                # Encode gpt response as label
                encoded_label = self.tokenizer(
                    gpt_text if gpt_text else "",
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                labels = encoded_label["input_ids"].squeeze(0)
                
                image_rel_path = Path(sample["image"])
                image_abs_path = self.image_root / image_rel_path
                
                # Store sample
                processed_sample = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "image_path": image_abs_path,
                    "idx": len(samples)
                }
                samples.append(processed_sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return processed sample with input_ids, labels, image and idx"""
        sample = self.samples[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            image_tensor = self.img_transform(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {sample['image_path']}, use zero tensor instead")
            image_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Image load failed {sample['image_path']}, error: {e}, use zero tensor instead")
            image_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
        
        return {
            "input_ids": sample["input_ids"].clone().detach(), 
            "labels": sample["labels"].clone().detach(),
            "image": image_tensor,
            "idx": torch.tensor(sample["idx"], dtype=torch.long) 
        }

def build_dataloader(
    data_path: str,
    tokenizer,
    image_root: str = "./",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    limit_size: int = -1,
    max_seq_len: int = 768,
    return_dataset: bool = False
):
    """Build dataloader for raw multimodal jsonl dataset"""
    # Initialize dataset
    dataset = MultimodalJsonlDataset(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        image_root=image_root,
        max_seq_len=max_seq_len
    )

    # Limit dataset size if required
    if limit_size > 0 and limit_size < len(dataset):
        dataset = Subset(dataset, list(range(limit_size)))
        print(f"Dataset size limited to: {limit_size} samples")

    # if return_dataset:
    #     return None, dataset

    # Custom collate function for batch assembly
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "image": torch.stack([item["image"] for item in batch]),
            "idx": torch.stack([item["idx"] for item in batch])
        }

    # Create and return dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True, 
        drop_last=False 
    )
    if return_dataset:
        return dataloader, dataset
    
    return dataloader, None