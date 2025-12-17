import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

IGNORE_INDEX = -100
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "<image>": 4,
    "<n>": 5,
}

def load_jsonl_data(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_image(image_path: str, image_root: Optional[str] = None) -> Image.Image:
    if image_root and not os.path.isabs(image_path):
        image_path = os.path.join(image_root, image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")

def dynamic_preprocess(
    image: Image.Image,
    image_size: int = 224,
    min_num: int = 1,
    max_num: int = 4,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    if use_thumbnail:
        imgs.append(image.resize((image_size, image_size), Image.Resampling.LANCZOS))
    n = random.randint(min_num, max_num)
    while len(imgs) < n:
        imgs.append(image.resize((image_size, image_size), Image.Resampling.LANCZOS))
    return imgs

class InstructionTuningDataset(Dataset):
    def __init__(
        self,
        data_path: Optional[str] = None, 
        data: Optional[List[Dict[str, Any]]] = None, 
        tokenizer=None,
        image_root: Optional[str] = None,
        image_size: int = 224,
        dynamic_image_size: bool = True,
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 2,
        use_thumbnail: bool = True,
        max_seq_len: int = 512,
    ):
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = load_jsonl_data(data_path)
        else:
            raise ValueError("Either 'data_path' or 'data' must be provided")

        self.tok = tokenizer
        self.image_root = image_root
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.max_seq_len = max_seq_len

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _get_image_path(image_info: Any) -> str:
        if isinstance(image_info, str):
            return image_info
        if isinstance(image_info, dict) and "path" in image_info:
            return image_info["path"]
        raise ValueError(f"Unsupported image field: {type(image_info)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        convs = item["conversations"]

        has_image = any("<image>" in c["value"] for c in convs)
        if not has_image:
            for c in convs:
                if c.get("from") == "human":
                    c["value"] = "<image>\n" + c["value"]
                    break

        text = ""
        for c in convs:
            role = c.get("from", "human")
            val = c.get("value", "")
            text += f"{role}: {val}\n"

        input_ids = self.tok.encode(text)[: self.max_seq_len]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        img_path = self._get_image_path(item["image"])
        img = load_image(img_path, self.image_root)

        if self.dynamic_image_size:
            imgs = dynamic_preprocess(
                img,
                image_size=self.image_size,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                use_thumbnail=self.use_thumbnail,
            )
        else:
            imgs = [img]

        pixel_values = torch.stack([self.transform(im) for im in imgs], dim=0)
        image_flags = torch.ones((pixel_values.size(0),), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "sample_index": torch.tensor(idx, dtype=torch.long),
            "original_index": torch.tensor(idx, dtype=torch.long),
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    max_len = max(x["input_ids"].size(0) for x in batch)
    max_patches = max(x["pixel_values"].size(0) for x in batch)
    _, _, H, W = batch[0]["pixel_values"].shape

    input_ids = torch.full((B, max_len), SPECIAL_TOKENS["<pad>"], dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)

    pixel_values = torch.zeros((B, max_patches, 3, H, W), dtype=torch.float32)
    image_flags = torch.zeros((B, max_patches), dtype=torch.long)

    sample_index = torch.zeros((B,), dtype=torch.long)
    original_index = torch.zeros((B,), dtype=torch.long) 

    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        P = x["pixel_values"].size(0)

        input_ids[i, :L] = x["input_ids"]
        attention_mask[i, :L] = 1
        labels[i, :L] = x["labels"]

        pixel_values[i, :P] = x["pixel_values"]
        image_flags[i, :P] = x["image_flags"]

        sample_index[i] = x["sample_index"]
        original_index[i] = x["original_index"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_flags": image_flags,
        "sample_index": sample_index,
        "original_index": original_index,
    }

def build_dataloader(
    data_path: Optional[str] = None,
    tokenizer=None,
    image_root: Optional[str] = None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    custom_dataset: Optional[Dataset] = None,
    return_dataset: bool = False,
    **dataset_kwargs
) -> Union[DataLoader, Tuple[DataLoader, Dataset]]:
    """
    Returns:
        DataLoader or (DataLoader, Dataset)
    """
    if custom_dataset is not None:
        ds = custom_dataset
    else:
        ds = InstructionTuningDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_root=image_root,
            **dataset_kwargs
        )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if return_dataset:
        return dl, ds
    return dl