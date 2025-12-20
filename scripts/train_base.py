import argparse
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from model import build_model
from data_utils_mnist import build_dataloader_mnist

def parse_args():
    parser = argparse.ArgumentParser("Train MNIST Base Model for Influence Function Calculation")
    # Data settings
    parser.add_argument("--mnist_root", type=str, default="./mnist_data")
    parser.add_argument("--train_limit", type=int, default=55000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100) 
    parser.add_argument("--lr", type=float, default=1.0) 
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "sgd", "adamw"])
    parser.add_argument("--lbfgs_max_iter", type=int, default=100) 
    parser.add_argument("--lbfgs_tolerance_grad", type=float, default=1e-8) 
    parser.add_argument("--lbfgs_tolerance_change", type=float, default=1e-10)  
    # Device & save
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >=0 and torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Build model
    model = build_model(
        experiment="mnist",
        weight_decay=args.weight_decay,
        is_multi=True
    ).to(device)

    # 2. Build dataset & dataloader
    _, train_dataset = build_dataloader_mnist(
        data_root=args.mnist_root,
        train=True,
        limit_size=args.train_limit,
        return_dataset=True
    )
    train_sample_num = len(train_dataset)
    print(f"Training sample number: {train_sample_num}")
    C = 1.0 / (train_sample_num * args.weight_decay)

    if args.optimizer == "lbfgs":
        batch_size = train_sample_num  # LBFGS uses full batch
        shuffle = False
    else:
        batch_size = args.batch_size
        shuffle = True

    train_loader, _ = build_dataloader_mnist(
        data_root=args.mnist_root,
        train=True,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        limit_size=args.train_limit
    )

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
            weight_decay=0.0
        )
    else:  # adamw
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.0
        )

    # 4. Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        if args.optimizer == "lbfgs":
            for batch in train_loader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)

                def closure():
                    optimizer.zero_grad()
                    logits = model(x, labels=y)
                    loss = model.loss(logits, y)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                total_loss = loss.item() * x.size(0)
        else:
            for batch in train_loader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = model(x, labels=y)
                loss = model.loss(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
        
        avg_loss = total_loss / train_sample_num
        print(f"Epoch [{epoch+1}/{args.epochs}], Avg Loss: {avg_loss:.6f}")

    # 5. Save model
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = save_dir / "base_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "weight_decay": args.weight_decay,
        "train_sample_num": train_sample_num,
        "C": C,
        "optimizer": args.optimizer
    }, ckpt_path)
    print(f"Base model saved to: {ckpt_path}")

if __name__ == "__main__":
    main()