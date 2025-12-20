import argparse
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from model import build_model, SimpleTokenizer
from data_utils_mnist import build_dataloader_mnist
from data_utils import build_dataloader

def parse_args():
    parser = argparse.ArgumentParser("Train Base Model (MNIST/Multimodal) for Influence Function Calculation")
    # Core: Experiment type selection (MNIST/Multimodal)
    parser.add_argument("--experiment", choices=["mnist", "multimodal"], required=True, help="Experiment type")
    
    # MNIST exclusive parameters
    parser.add_argument("--data_root", type=str, default="./mnist_data", help="MNIST data root directory")
    
    # Multimodal exclusive parameters (match bash script config)
    parser.add_argument("--train_jsonl", type=str, default=None, help="Path to multimodal train jsonl file")
    parser.add_argument("--test_jsonl", type=str, default=None, help="Path to multimodal test jsonl file")
    parser.add_argument("--image_root", type=str, default="./", help="Root directory for multimodal images")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension of multimodal model")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads in multimodal model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers in multimodal model")
    parser.add_argument("--max_seq_len", type=int, default=768, help="Max sequence length for multimodal text")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocab size of multimodal tokenizer")
    
    # Common training parameters (for both experiments)
    parser.add_argument("--train_limit", type=int, default=55000, help="Limit training data size (-1 for full dataset)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size (ignored for LBFGS full batch)")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization)")
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "sgd", "adamw"], help="Optimizer type")
    parser.add_argument("--lbfgs_max_iter", type=int, default=100, help="Max iterations for LBFGS optimizer")
    parser.add_argument("--lbfgs_tolerance_grad", type=float, default=1e-8, help="Gradient tolerance for LBFGS convergence")
    parser.add_argument("--lbfgs_tolerance_change", type=float, default=1e-10, help="Parameter change tolerance for LBFGS convergence")
    
    # Device and save parameters
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the trained base model")
    return parser.parse_args()

def main():
    args = parse_args()
    # Set training device (GPU/CPU)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Experiment type: {args.experiment}")

    # 1. Build model
    if args.experiment == "mnist":
        model = build_model(
            experiment="mnist",
            weight_decay=args.weight_decay,
            is_multi=True
        ).to(device)
        tokenizer = None 
    else:  # multimodal
        tokenizer = SimpleTokenizer(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len
        )
        # Build multimodal transformer model
        model = build_model(
            experiment="multimodal",
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            weight_decay=args.weight_decay  # Add weight_decay for model
        ).to(device)
        # Validate multimodal data paths
        if not args.train_jsonl or not args.test_jsonl:
            raise ValueError("Multimodal experiment requires --train_jsonl and --test_jsonl parameters")

    # 2. Build dataset and dataloader
    if args.experiment == "mnist":
        # Load MNIST training dataset
        _, train_dataset = build_dataloader_mnist(
            data_root=args.data_root,
            train=True,
            limit_size=args.train_limit,
            return_dataset=True
        )
        train_sample_num = len(train_dataset)
        print(f"MNIST training sample number: {train_sample_num}")
        
        # Set batch size (full batch for LBFGS, specified batch for others)
        if args.optimizer == "lbfgs":
            batch_size = train_sample_num
            shuffle = False
        else:
            batch_size = args.batch_size
            shuffle = True
        
        # Build MNIST training dataloader
        train_loader, _ = build_dataloader_mnist(
            data_root=args.data_root,
            train=True,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            limit_size=args.train_limit
        )
    else:  # multimodal
        # Load multimodal training dataset
        _, train_dataset = build_dataloader(
            data_path=args.train_jsonl,
            tokenizer=tokenizer,
            image_root=args.image_root,
            max_seq_len=args.max_seq_len, 
            return_dataset=True
        )
        train_sample_num = len(train_dataset)
        print(f"Multimodal training sample number: {train_sample_num}")
        
        # Set batch size (full batch for LBFGS, specified batch for others)
        batch_size = args.batch_size
        shuffle = False
        
        # Build multimodal training dataloader
        train_loader, _ = build_dataloader(
            data_path=args.train_jsonl,
            tokenizer=tokenizer,
            image_root=args.image_root,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            max_seq_len=args.max_seq_len,  # Pass max_seq_len
            return_dataset=False
        )

    # Calculate regularization coefficient C
    C = 1.0 / (train_sample_num * args.weight_decay)

    # 3. Build optimizer
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
            weight_decay=0.0  # L2 regularization implemented in model
        )
    else:  # adamw
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.0  # L2 regularization implemented in model
        )

    print(f"optimizer:{args.optimizer}")

    # 4. Training loop (compatible with both data formats)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        if args.optimizer == "lbfgs":
            # LBFGS training with closure function
            for batch in train_loader:
                # Process different batch formats
                if args.experiment == "mnist":
                    x, y, _ = batch
                    x, y = x.to(device), y.to(device)
                else:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    input_ids = batch["input_ids"]
                    pixel_values = batch["image"]  # Align with model's pixel_values
                    y = batch["labels"]
                    # Generate attention mask (non-pad tokens are 1)
                    attention_mask = (input_ids != tokenizer.pad_token_id).float()

                def closure():
                    optimizer.zero_grad()
                    if args.experiment == "mnist":
                        logits = model(x, labels=y)
                        loss = model.loss(logits, y)
                    else:
                        # Pass all required parameters to multimodal model
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=y
                        )
                        loss = model.loss(logits, y) if y is None else logits  # Model returns loss directly when labels are passed
                    loss.backward()
                    return loss

                # Execute LBFGS optimization
                loss = optimizer.step(closure)
                batch_size = x.size(0) if args.experiment == "mnist" else input_ids.size(0)
                total_loss = loss.item() * batch_size
        else:
            # SGD/AdamW normal training loop
            for batch in train_loader:
                # Process different batch formats
                if args.experiment == "mnist":
                    x, y, _ = batch
                    x, y = x.to(device), y.to(device)
                else:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    input_ids = batch["input_ids"]
                    pixel_values = batch["image"]  # Align with model's pixel_values
                    y = batch["labels"]
                    # Generate attention mask (non-pad tokens are 1)
                    attention_mask = (input_ids != tokenizer.pad_token_id).float()

                optimizer.zero_grad()
                if args.experiment == "mnist":
                    logits = model(x, labels=y)
                    loss = model.loss(logits, y)
                else:
                    # Pass all required parameters to multimodal model
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=y
                    )
                    loss = model.loss(logits, y) if y is None else logits  # Model returns loss directly when labels are passed
                loss.backward()
                optimizer.step()

                batch_size = x.size(0) if args.experiment == "mnist" else input_ids.size(0)
                total_loss += loss.item() * batch_size
        
        # Calculate and print average loss
        avg_loss = total_loss / train_sample_num
        print(f"Epoch [{epoch+1}/{args.epochs}], Avg Loss: {avg_loss:.6f}")

    # 5. Save model (save extra config for multimodal)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = save_dir / "base_model.pth"

    # Build checkpoint dictionary
    ckpt_dict = {
        "model_state_dict": model.state_dict(),
        "weight_decay": args.weight_decay,
        "train_sample_num": train_sample_num,
        "C": C,
        "optimizer": args.optimizer,
        "experiment": args.experiment
    }

    # Add tokenizer and model config for multimodal
    if args.experiment == "multimodal":
        ckpt_dict["tokenizer_config"] = {
            "vocab_size": args.vocab_size,
            "max_seq_len": args.max_seq_len
        }
        ckpt_dict["model_config"] = {
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers
        }

    # Save checkpoint
    torch.save(ckpt_dict, ckpt_path)
    print(f"Base model saved to: {ckpt_path}")
    if args.experiment == "multimodal":
        print("Tokenizer and model config saved with checkpoint")

if __name__ == "__main__":
    main()