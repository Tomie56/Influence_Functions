import os
import argparse
import logging
import torch

from run_loo import (
    train_mnist_lr,
    train_multimodal,
    build_mnist_datasets,
    build_multimodal_datasets,
    get_device,
    setup_logging,
)


def train_mnist(args, device: torch.device):
    train_ds, _ = build_mnist_datasets(args, device)

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
    return model


def train_multimodal_exp(args, device: torch.device):
    if not args.train_jsonl:
        raise ValueError("--train_jsonl is required for experiment=multimodal")

    if not args.test_jsonl:
        raise ValueError("--test_jsonl is required for experiment=multimodal (dataset builder needs it)")

    train_ds, _ = build_multimodal_datasets(args, device)

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
    return model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train base model and save checkpoint")

    p.add_argument("--experiment", choices=["mnist", "multimodal"], required=True)
    p.add_argument("--save_path", type=str, required=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)

    # MNIST
    p.add_argument("--mnist_root", type=str, default="./mnist")
    p.add_argument("--mnist_train_limit", type=int, default=55000)
    p.add_argument("--mnist_test_limit", type=int, default=10000)  # dataset builder needs it
    p.add_argument("--optimizer", choices=["lbfgs", "adamw", "sgd"], default="lbfgs")
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


def main():
    setup_logging()
    args = build_parser().parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    device = get_device()

    logging.info(f"Device: {device}")
    logging.info(f"Experiment: {args.experiment}")
    logging.info(f"Saving to: {args.save_path}")

    if args.experiment == "mnist":
        model = train_mnist(args, device)
    else:
        model = train_multimodal_exp(args, device)

    torch.save(model.state_dict(), args.save_path)
    logging.info(f"Saved checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()
