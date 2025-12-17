import os
import json
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
    )

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_done_ids(path: str) -> Set[int]:
    done: Set[int] = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                if "id" in j:
                    done.add(int(j["id"]))
            except Exception:
                continue
    return done

def read_ids_txt(path: str) -> List[int]:
    ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(int(s))
    return ids

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Dataset wrapper: exclude one original index without building big lists
# ------------------------------------------------------------

class ExcludeOneIndexDataset(Dataset):
    def __init__(self, base: Dataset, exclude_idx: int):
        self.base = base
        self.exclude_idx = int(exclude_idx)
        self.n = len(base)
        if self.exclude_idx < 0 or self.exclude_idx >= self.n:
            raise IndexError(f"exclude_idx out of range: {self.exclude_idx} / {self.n}")

    def __len__(self) -> int:
        return self.n - 1

    def __getitem__(self, i: int):
        if i < 0 or i >= self.n - 1:
            raise IndexError(i)
        j = i if i < self.exclude_idx else i + 1
        return self.base[j]

# ------------------------------------------------------------
# Loss adapters (mnist vs multimodal)
# ------------------------------------------------------------

def _is_mnist_item(item: Any) -> bool:
    if isinstance(item, dict):
        return ("x" in item) and ("labels" in item)
    if isinstance(item, (tuple, list)):
        return len(item) >= 2
    return False

def _to_1d_long_tensor(v: Any) -> torch.Tensor:
    if torch.is_tensor(v):
        return v.view(-1).long()
    return torch.tensor([int(v)], dtype=torch.long)


def _extract_mnist(item: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    统一解析 MNIST item，返回 (x, y, idx) 且 y/idx 都是 1D long tensor
    - dict: {"x","labels", optional "idx"}
    - tuple/list: (x, y) or (x, y, idx)
    """
    if isinstance(item, dict):
        x = item["x"]
        y = _to_1d_long_tensor(item["labels"])
        idx = _to_1d_long_tensor(item.get("idx", -1))
        return x, y, idx

    if isinstance(item, (tuple, list)) and len(item) >= 2:
        x = item[0]
        y = _to_1d_long_tensor(item[1])
        idx = _to_1d_long_tensor(item[2] if len(item) >= 3 else -1)
        return x, y, idx

    raise TypeError(f"Unknown MNIST item type: {type(item)}")



def _is_multimodal_item(item: Dict[str, Any]) -> bool:
    return isinstance(item, dict) and ("input_ids" in item) and ("labels" in item)

@torch.no_grad()
def compute_loss_on_test(model: nn.Module, test_item: Dict[str, Any], device: torch.device) -> float:
    model.eval()

    if _is_mnist_item(test_item):
        x, y, _ = _extract_mnist(test_item)
        x = x.unsqueeze(0).to(device)
        y = y.view(-1).long().to(device)
        out = model(x=x, labels=y)
        return float(out.loss.item())

    if _is_multimodal_item(test_item):
        batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in test_item.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_flags=batch.get("image_flags"),
            labels=batch.get("labels"),
        )
        if out.loss is None:
            raise RuntimeError("multimodal model returned loss=None on test")
        return float(out.loss.item())

    raise KeyError(f"Unknown test_item keys: {list(test_item.keys())}")

# ------------------------------------------------------------
# MNIST training (TorchLogisticRegression)
# ------------------------------------------------------------

def _mnist_collate(batch: List[Any]) -> Dict[str, torch.Tensor]:
    xs, ys, idxs = [], [], []
    for it in batch:
        x, y, idx = _extract_mnist(it)
        xs.append(x)
        ys.append(y[0]) 
        idxs.append(idx[0])

    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0).long().view(-1)
    idxs = torch.stack(idxs, dim=0).long().view(-1)
    return {"x": xs, "labels": ys, "idx": idxs}


def train_mnist_lr(
    train_subset: Dataset,
    device: torch.device,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    l2_reg: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> nn.Module:
    from model import TorchLogisticRegression, MNIST_INPUT_DIM, MNIST_NUM_CLASSES

    model = TorchLogisticRegression(
        input_dim=MNIST_INPUT_DIM,
        num_classes=MNIST_NUM_CLASSES,
        loss_reduction="mean",
        l2_reg=float(l2_reg),
    ).to(device)

    pin_memory = (device.type == "cuda")
    optimizer_name = optimizer_name.lower().strip()

    if optimizer_name == "lbfgs":
        # LBFGS 
        loader = DataLoader(
            train_subset,
            batch_size=len(train_subset),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_mnist_collate,
        )
        opt = optim.LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=100, line_search_fn="strong_wolfe")

        for ep in range(epochs):
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)

                def closure():
                    opt.zero_grad(set_to_none=True)
                    out = model(x=x, labels=y)
                    loss = out.loss
                    loss.backward()
                    return loss

                loss = opt.step(closure)
            logging.info(f"[MNIST/LBFGS] epoch {ep+1}/{epochs} loss={float(loss.item()):.6f}")

        return model

    # If use Adam / SGD 
    loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_mnist_collate,
    )

    if optimizer_name == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = model(x=x, labels=y)
            loss = out.loss
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1
        logging.info(f"[MNIST/{optimizer_name}] epoch {ep+1}/{epochs} avg_loss={total/max(n,1):.6f}")

    return model

# ------------------------------------------------------------
# Multimodal training
# ------------------------------------------------------------

def _multimodal_forward_loss(model: nn.Module, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    b = {}
    for k, v in batch.items():
        b[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v

    out = model(
        input_ids=b["input_ids"],
        attention_mask=b.get("attention_mask"),
        pixel_values=b.get("pixel_values"),
        image_flags=b.get("image_flags"),
        labels=b.get("labels"),
    )
    if out.loss is None:
        raise RuntimeError("multimodal model returned loss=None in training")
    return out.loss

def train_multimodal(
    train_subset: Dataset,
    device: torch.device,
    image_root: str,
    image_size: int,
    max_seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    num_image_tokens_per_patch: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
) -> nn.Module:
    from model import MiniMultimodalModel, DEFAULT_VOCAB_SIZE, DEFAULT_IMAGE_EMBED_DIM

    model = MiniMultimodalModel(
        vocab_size=DEFAULT_VOCAB_SIZE,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        image_embed_dim=DEFAULT_IMAGE_EMBED_DIM,
        image_size=image_size,
        num_image_tokens_per_patch=num_image_tokens_per_patch,
        max_seq_len=max_seq_len,
    ).to(device)

    pin_memory = (device.type == "cuda")
    loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = _multimodal_forward_loss(model, batch, device)
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1
        logging.info(f"[MULTIMODAL] epoch {ep+1}/{epochs} avg_loss={total/max(n,1):.6f}")

    return model

# ------------------------------------------------------------
# Build datasets
# ------------------------------------------------------------

def build_mnist_datasets(args, device: torch.device) -> Tuple[Dataset, Dataset]:
    from data_utils_mnist import build_dataloader_mnist

    _, train_ds = build_dataloader_mnist(
        data_root=args.mnist_root,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        download=True,
        normalize=True,
        return_dataset=True,
        limit_size=args.mnist_train_limit,
    )

    _, test_ds = build_dataloader_mnist(
        data_root=args.mnist_root,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        download=True,
        normalize=True,
        return_dataset=True,
        limit_size=args.mnist_test_limit,
    )
    return train_ds, test_ds

def build_multimodal_datasets(args, device: torch.device) -> Tuple[Dataset, Dataset]:
    from model import SimpleTokenizer, DEFAULT_VOCAB_SIZE
    from data_utils import build_dataloader

    tokenizer = SimpleTokenizer(vocab_size=DEFAULT_VOCAB_SIZE, max_seq_len=args.max_seq_len)

    dataset_kwargs = dict(
        image_size=args.image_size,
        dynamic_image_size=True,
        min_dynamic_patch=1,
        max_dynamic_patch=2,
        use_thumbnail=True,
        max_seq_len=args.max_seq_len,
    )

    _, train_ds = build_dataloader(
        data_path=args.train_jsonl,
        tokenizer=tokenizer,
        image_root=args.image_root,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        return_dataset=True,
        **dataset_kwargs,
    )

    _, test_ds = build_dataloader(
        data_path=args.test_jsonl,
        tokenizer=tokenizer,
        image_root=args.image_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        return_dataset=True,
        **dataset_kwargs,
    )
    return train_ds, test_ds

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Run Leave-One-Out retraining and log per-removal test loss to jsonl")

    p.add_argument("--experiment", choices=["mnist", "multimodal"], required=True)

    p.add_argument("--topk_ids_txt", type=str, required=True)
    p.add_argument("--test_id", type=int, required=True)
    p.add_argument("--save_jsonl", type=str, required=True)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)

    # MNIST
    p.add_argument("--mnist_root", type=str, default="./mnist")
    p.add_argument("--mnist_train_limit", type=int, default=55000)
    p.add_argument("--mnist_test_limit", type=int, default=10000)
    p.add_argument("--optimizer", type=str, default="lbfgs")   # lbfgs | adamw | sgd
    p.add_argument("--l2_reg", type=float, default=0.01)

    # Multimodal
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--test_jsonl", type=str, default=None)
    p.add_argument("--image_root", type=str, default="./")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_image_tokens_per_patch", type=int, default=12)
    p.add_argument("--max_seq_len", type=int, default=768)

    return p

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    device = get_device()

    ensure_dir(args.save_jsonl)

    topk_ids = read_ids_txt(args.topk_ids_txt)
    done_ids = load_done_ids(args.save_jsonl)

    logging.info(f"Device: {device}")
    logging.info(f"Experiment: {args.experiment}")
    logging.info(f"Test id: {args.test_id}")
    logging.info(f"TopK ids: {len(topk_ids)} (resume done: {len(done_ids)})")
    logging.info(f"Save jsonl: {args.save_jsonl}")

    # Build datasets
    if args.experiment == "mnist":
        train_ds, test_ds = build_mnist_datasets(args, device)
    else:
        if not args.train_jsonl or not args.test_jsonl:
            raise ValueError("multimodal requires --train_jsonl and --test_jsonl")
        train_ds, test_ds = build_multimodal_datasets(args, device)

    if args.test_id < 0 or args.test_id >= len(test_ds):
        raise IndexError(f"test_id out of range: {args.test_id} / {len(test_ds)}")
    test_item = test_ds[args.test_id]

    # ------------------------------------------------------------
    # (0) Base loss (no removal): write {"id": -1, "loss": ...}
    # ------------------------------------------------------------
    if -1 not in done_ids:
        logging.info("[BASE] Training on full train set to get base loss (id=-1)")

        if args.experiment == "mnist":
            model = train_mnist_lr(
                train_subset=train_ds,
                device=device,
                optimizer_name=args.optimizer,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                l2_reg=float(args.l2_reg),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
        else:
            model = train_multimodal(
                train_subset=train_ds,
                device=device,
                image_root=args.image_root,
                image_size=int(args.image_size),
                max_seq_len=int(args.max_seq_len),
                embed_dim=int(args.embed_dim),
                num_heads=int(args.num_heads),
                num_layers=int(args.num_layers),
                num_image_tokens_per_patch=int(args.num_image_tokens_per_patch),
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )

        base_loss = compute_loss_on_test(model, test_item, device=device)
        append_jsonl(args.save_jsonl, {"id": -1, "loss": float(base_loss)})
        logging.info(f"[BASE] id=-1 loss={base_loss:.6f} appended")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logging.info("[BASE] Found id=-1 in jsonl, skip base loss")

    # ------------------------------------------------------------
    # (1) LOO loop: for each train_id, remove it and retrain
    # ------------------------------------------------------------
    n_train = len(train_ds)

    for j, train_id in enumerate(topk_ids, start=1):
        train_id = int(train_id)
        if train_id in done_ids:
            logging.info(f"[LOO {j}/{len(topk_ids)}] skip train_id={train_id} (already in jsonl)")
            continue
        if train_id < 0 or train_id >= n_train:
            logging.warning(f"[LOO {j}/{len(topk_ids)}] skip invalid train_id={train_id} (train size={n_train})")
            continue

        logging.info(f"[LOO {j}/{len(topk_ids)}] train_id={train_id} (remove + retrain)")

        loo_train_ds = ExcludeOneIndexDataset(train_ds, exclude_idx=train_id)

        if args.experiment == "mnist":
            model = train_mnist_lr(
                train_subset=loo_train_ds,
                device=device,
                optimizer_name=args.optimizer,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                l2_reg=float(args.l2_reg),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
        else:
            model = train_multimodal(
                train_subset=loo_train_ds,
                device=device,
                image_root=args.image_root,
                image_size=int(args.image_size),
                max_seq_len=int(args.max_seq_len),
                embed_dim=int(args.embed_dim),
                num_heads=int(args.num_heads),
                num_layers=int(args.num_layers),
                num_image_tokens_per_patch=int(args.num_image_tokens_per_patch),
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )

        loo_loss = compute_loss_on_test(model, test_item, device=device)
        append_jsonl(args.save_jsonl, {"id": train_id, "loss": float(loo_loss)})
        logging.info(f"[LOO {j}/{len(topk_ids)}] train_id={train_id} loss={loo_loss:.6f} appended")

        done_ids.add(train_id)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info("DONE")


if __name__ == "__main__":
    main()
