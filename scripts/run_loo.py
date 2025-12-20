import argparse
import torch
import torch.optim as optim
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import build_model, SimpleTokenizer
from data_utils_mnist import build_dataloader_mnist, mnist_collate_fn
from data_utils import build_dataloader

# -------------------------- Utility Functions --------------------------
class ExcludeOneDataset(Dataset):
    """Dataset that excludes a single specified training sample by index"""
    def __init__(self, base_dataset: Dataset, exclude_idx: int):
        self.base_dataset = base_dataset
        self.exclude_idx = exclude_idx
        self.valid_indices = [i for i in range(len(base_dataset)) if i != exclude_idx]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.valid_indices[idx]]

def load_done_ids(jsonl_path: Path) -> set:
    """Load completed LOO training sample IDs for resume functionality"""
    done_ids = set()
    if not jsonl_path.exists():
        return done_ids
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                done_ids.add(int(data["id"]))
            except Exception as e:
                continue
    return done_ids

def read_topk_ids(txt_path: Path) -> list:
    """Read Top-K influential training sample IDs from txt file"""
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(line))
    return ids

def append_loo_result(jsonl_path: Path, data: dict):
    """Append LOO retraining result to jsonl file (line-by-line JSON)"""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# -------------------------- Model Training & Loss Calculation --------------------------
def train_loo_model(args, train_dataset: Dataset, exclude_idx: int, device, tokenizer=None):
    """Train model with one specific training sample excluded (Leave-One-Out)"""
    # Construct LOO training dataset (exclude target sample)
    loo_train_dataset = ExcludeOneDataset(train_dataset, exclude_idx)
    
    # Build model based on experiment type
    if args.experiment == "mnist":
        model = build_model(
            experiment="mnist",
            weight_decay=args.weight_decay,
            is_multi=True  # Align with MNIST multi-class task
        ).to(device)
        # Determine batch size (full dataset for LBFGS, specified batch size for other optimizers)
        batch_size = args.batch_size if args.optimizer != "lbfgs" else len(loo_train_dataset)
        train_loader = DataLoader(
            loo_train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=mnist_collate_fn
        )
    else:
        model = build_model(
            experiment="multimodal",
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len
        ).to(device)
        # Build dataloader for multimodal dataset
        train_loader = DataLoader(
            loo_train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    # Build optimizer (align with PyTorch LBFGS parameters)
    if args.optimizer == "lbfgs":
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            history_size=20,
            line_search_fn="strong_wolfe"
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0  # L2 regularization implemented inside model
        )
    else:  # adamw
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.0  # L2 regularization implemented inside model
        )

    # Training loop (adapt to LBFGS closure requirement)
    model.train()
    for epoch in range(args.epochs):
        if args.optimizer == "lbfgs":
            for batch in train_loader:
                # Move batch to target device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    x = batch["input_ids"]
                    y = batch["labels"]
                else:
                    x, y, _ = batch
                    x, y = x.to(device), y.to(device)

                # Closure for LBFGS (required to compute loss and gradients)
                def closure():
                    optimizer.zero_grad()
                    logits = model(x, labels=y)
                    loss = model.loss(logits, y)
                    loss.backward()
                    return loss
                optimizer.step(closure)
        else:
            # Standard training loop for SGD/AdamW
            for batch in train_loader:
                # Move batch to target device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    x = batch["input_ids"]
                    y = batch["labels"]
                else:
                    x, y, _ = batch
                    x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = model(x, labels=y)
                loss = model.loss(logits, y)
                loss.backward()
                optimizer.step()

    return model.eval()

def compute_test_loss(args, model, test_dataset: Dataset, test_id: int, device):
    """Compute test loss for the specified target test sample"""
    # Get single test sample
    test_sample = test_dataset[test_id]
    if isinstance(test_sample, dict):
        x_test = test_sample["input_ids"].unsqueeze(0).to(device)
        y_test = test_sample["labels"].unsqueeze(0).to(device)
    else:
        x_test, y_test, _ = test_sample
        
        x_test = x_test.unsqueeze(0).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).unsqueeze(0).to(device)
        # x_test, y_test = x_test.unsqueeze(0).to(device), y_test.unsqueeze(0).to(device)

    # Calculate loss without gradient computation
    with torch.no_grad():
        logits = model(x_test, labels=y_test)
        loss = model.loss(logits, y_test)
    return loss.item()

# -------------------------- Main Logic --------------------------
def parse_args():
    parser = argparse.ArgumentParser("Leave-One-Out (LOO) Retraining with Resume Support")
    # Experiment configuration
    parser.add_argument("--experiment", choices=["mnist", "multimodal"], required=True, help="Experiment type (mnist/multimodal)")
    # Data & ID configuration
    parser.add_argument("--topk_ids_txt", type=str, required=True, help="Path to Top-K influential train IDs txt file")
    parser.add_argument("--test_id", type=int, required=True, help="Target test sample ID for loss evaluation")
    # Data paths
    parser.add_argument("--data_root", type=str, default="./mnist", help="dataset root directory")
    parser.add_argument("--train_limit", type=int, default=55000, help="Limit size of training dataset")
    parser.add_argument("--test_limit", type=int, default=10000, help="Limit size of test dataset")
    parser.add_argument("--train_jsonl", type=str, default=None, help="Path to multimodal train jsonl file")
    parser.add_argument("--test_jsonl", type=str, default=None, help="Path to multimodal test jsonl file")
    parser.add_argument("--image_root", type=str, default="./", help="Root directory for multimodal images")
    # Model configuration
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension for multimodal model")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads for multimodal model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers for multimodal model")
    parser.add_argument("--max_seq_len", type=int, default=768, help="Max sequence length for multimodal text")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization) coefficient")
    # Training configuration
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "sgd", "adamw"], help="Optimizer type")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (ignored for LBFGS)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    # LBFGS specific parameters
    parser.add_argument("--lbfgs_max_iter", type=int, default=20, help="Max iterations for LBFGS optimizer")
    parser.add_argument("--lbfgs_tolerance_grad", type=float, default=1e-8, help="Gradient tolerance for LBFGS convergence")
    parser.add_argument("--lbfgs_tolerance_change", type=float, default=1e-10, help="Parameter change tolerance for LBFGS convergence")
    # Output configuration
    parser.add_argument("--save_jsonl", type=str, required=True, help="Path to save LOO results (jsonl format)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)")
    return parser.parse_args()

def build_base_datasets(args, tokenizer=None):
    """Build base training and test datasets (without sample exclusion)"""
    if args.experiment == "mnist":
        _, train_dataset = build_dataloader_mnist(
            data_root=args.data_root,
            train=True,
            limit_size=args.train_limit,
            return_dataset=True
        )
        _, test_dataset = build_dataloader_mnist(
            data_root=args.data_root,
            train=False,
            limit_size=args.test_limit,
            return_dataset=True
        )
    else:
        _, train_dataset = build_dataloader(
            data_path=args.train_jsonl,
            tokenizer=tokenizer,
            image_root=args.image_root,
            return_dataset=True
        )
        _, test_dataset = build_dataloader(
            data_path=args.test_jsonl,
            tokenizer=tokenizer,
            image_root=args.image_root,
            return_dataset=True
        )
    return train_dataset, test_dataset

def main():
    args = parse_args()
    # Set device (GPU/CPU)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer (only for multimodal experiment)
    tokenizer = None
    if args.experiment == "multimodal":
        tokenizer = SimpleTokenizer(max_seq_len=args.max_seq_len)

    # Build base datasets
    train_dataset, test_dataset = build_base_datasets(args, tokenizer)

    # Load Top-K IDs and completed LOO IDs
    topk_ids_path = Path(args.topk_ids_txt)
    jsonl_path = Path(args.save_jsonl)
    topk_ids = read_topk_ids(topk_ids_path)
    done_ids = load_done_ids(jsonl_path)
    print(f"Loaded {len(topk_ids)} Top-K IDs, {len(done_ids)} already completed")

    # Train base model (no sample excluded, ID=-1) if not completed
    if -1 not in done_ids:
        print("Training base model")
        base_model = train_loo_model(args, train_dataset, exclude_idx=-1, device=device, tokenizer=tokenizer)
        base_loss = compute_test_loss(args, base_model, test_dataset, args.test_id, device)
        append_loo_result(jsonl_path, {"id": -1, "loss": base_loss})
        done_ids.add(-1)
        print(f"Base model loss: {base_loss:.6f} ")
        del base_model
        torch.cuda.empty_cache()

    # Perform LOO retraining for Top-K samples
    for train_id in tqdm(topk_ids, desc="LOO Retraining"):
        if train_id in done_ids:
            continue
        if train_id < 0 or train_id >= len(train_dataset):
            print(f"Invalid train ID: {train_id} (skipped)")
            continue

        try:
            # Train LOO model (exclude current train sample)
            loo_model = train_loo_model(args, train_dataset, exclude_idx=train_id, device=device, tokenizer=tokenizer)
            # Compute test loss for target test sample
            loo_loss = compute_test_loss(args, loo_model, test_dataset, args.test_id, device)
            # Save LOO result
            append_loo_result(jsonl_path, {"id": train_id, "loss": loo_loss})
            done_ids.add(train_id)
            print(f"Train ID {train_id} - LOO loss: {loo_loss:.6f} ")
        except Exception as e:
            print(f"Train ID {train_id} - training failed: {str(e)}")
        finally:
            # Release GPU memory
            del loo_model
            torch.cuda.empty_cache()

    print("LOO retraining completed!")

if __name__ == "__main__":
    main()