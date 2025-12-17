import os
import json
import argparse
import logging
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from influence_function import calculate_influences


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_topk_ids_txt(path: str, ids: List[int]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(f"{int(i)}\n")


def save_scores_npy(path: str, scores: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.save(path, scores.astype(np.float32))


def topk_by_abs(scores: np.ndarray, k: int) -> List[int]:
    k = int(min(max(k, 0), scores.shape[0]))
    if k == 0:
        return []
    idx = np.argsort(np.abs(scores))[-k:][::-1]
    return idx.astype(int).tolist()


def load_base_model_mnist(ckpt_path: str, device: torch.device):
    from model import TorchLogisticRegression, MNIST_INPUT_DIM, MNIST_NUM_CLASSES
    model = TorchLogisticRegression(
        input_dim=MNIST_INPUT_DIM,
        num_classes=MNIST_NUM_CLASSES,
        loss_reduction="mean",
        l2_reg=0.0,
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def load_base_model_multimodal(ckpt_path: str, device: torch.device, args):
    from model import MiniMultimodalModel, DEFAULT_VOCAB_SIZE, DEFAULT_IMAGE_EMBED_DIM
    model = MiniMultimodalModel(
        vocab_size=DEFAULT_VOCAB_SIZE,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        image_embed_dim=DEFAULT_IMAGE_EMBED_DIM,
        image_size=args.image_size,
        num_image_tokens_per_patch=args.num_image_tokens_per_patch,
        max_seq_len=args.max_seq_len,
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def build_loaders_mnist(args, device: torch.device):
    from data_utils_mnist import build_dataloader_mnist

    train_loader, train_dataset = build_dataloader_mnist(
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

    test_loader, test_dataset = build_dataloader_mnist(
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

    return train_loader, train_dataset, test_loader, test_dataset


def build_loaders_multimodal(args, device: torch.device):
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

    train_loader, train_dataset = build_dataloader(
        data_path=args.train_jsonl,
        tokenizer=tokenizer,
        image_root=args.image_root,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        return_dataset=True,
        **dataset_kwargs,
    )

    test_loader, test_dataset = build_dataloader(
        data_path=args.test_jsonl,
        tokenizer=tokenizer,
        image_root=args.image_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        return_dataset=True,
        **dataset_kwargs,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def _one_sample_loader_from_dataset(
    dataset,
    sample_index: int,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    subset = Subset(dataset, [int(sample_index)])
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def run_single_test(args, device: torch.device) -> None:
    ensure_dir(args.tmp_dir)

    if args.experiment == "mnist":
        train_loader, _, _, test_dataset = build_loaders_mnist(args, device)
        model = load_base_model_mnist(args.ckpt_path, device)
    else:
        train_loader, _, _, test_dataset = build_loaders_multimodal(args, device)
        model = load_base_model_multimodal(args.ckpt_path, device, args)

    if args.test_id < 0 or args.test_id >= len(test_dataset):
        raise IndexError(f"--test_id out of range: {args.test_id} (test size={len(test_dataset)})")

    test_loader = _one_sample_loader_from_dataset(
        dataset=test_dataset,
        sample_index=args.test_id,
        device=device,
        num_workers=args.num_workers,
    )

    influences, _, _ = calculate_influences(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_test_samples=1,
        recursion_depth=args.recursion_depth,
        damp=args.damp,
        scale=args.scale,
        save_intermediate=False,
        output_dir=args.tmp_dir,
        top_k=0,
    )

    if len(influences) != 1:
        raise RuntimeError(f"Expected exactly 1 test in influences, got keys={list(influences.keys())}")

    test_key = next(iter(influences.keys()))
    scores = np.asarray(influences[test_key], dtype=np.float32)

    if args.save_scores_npy:
        save_scores_npy(args.save_scores_npy, scores)
        logging.info(f"Saved IF scores: {args.save_scores_npy} (len={len(scores)})")

    if args.save_topk_ids_txt:
        topk_ids = topk_by_abs(scores, args.topk)
        write_topk_ids_txt(args.save_topk_ids_txt, topk_ids)
        logging.info(f"Saved TopK ids: {args.save_topk_ids_txt} (k={len(topk_ids)})")

    logging.info(f"[SINGLE] done. test_key={test_key} (requested test_id={args.test_id})")


def run_all_tests(args, device: torch.device) -> None:
    if not args.save_all_dir:
        raise ValueError("--save_all_dir is required when --all_test_ids=1")

    ensure_dir(args.save_all_dir)
    ensure_dir(args.tmp_dir)

    index_path = os.path.join(args.save_all_dir, "index.jsonl")

    if args.experiment == "mnist":
        train_loader, _, _, test_dataset = build_loaders_mnist(args, device)
        model = load_base_model_mnist(args.ckpt_path, device)
    else:
        train_loader, _, _, test_dataset = build_loaders_multimodal(args, device)
        model = load_base_model_multimodal(args.ckpt_path, device, args)

    n = len(test_dataset)
    limit = args.all_test_limit if args.all_test_limit and args.all_test_limit > 0 else n
    limit = min(limit, n)
    test_ids = list(range(limit))

    logging.info(f"[ALL] test size={n}, will run={len(test_ids)}")
    logging.info(f"[ALL] save_dir={args.save_all_dir} (one npy per test)")

    with open(index_path, "a", encoding="utf-8") as fidx:
        for tid in test_ids:
            out_path = os.path.join(args.save_all_dir, f"if_scores_test_{tid}.npy")
            if args.resume_all_test and os.path.exists(out_path):
                logging.info(f"[ALL] skip existing test_id={tid}")
                continue

            test_loader = _one_sample_loader_from_dataset(
                dataset=test_dataset,
                sample_index=tid,
                device=device,
                num_workers=args.num_workers,
            )

            influences, _, _ = calculate_influences(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                num_test_samples=1,
                recursion_depth=args.recursion_depth,
                damp=args.damp,
                scale=args.scale,
                save_intermediate=False,
                output_dir=args.tmp_dir,
                top_k=0,
            )

            if len(influences) != 1:
                raise RuntimeError(f"[ALL] Expected exactly 1 test for tid={tid}, got keys={list(influences.keys())}")

            test_key = next(iter(influences.keys()))
            scores = np.asarray(influences[test_key], dtype=np.float32)
            save_scores_npy(out_path, scores)

            fidx.write(json.dumps({"test_id": int(tid), "test_key": int(test_key), "path": out_path}, ensure_ascii=False) + "\n")
            fidx.flush()

            logging.info(f"[ALL] saved test_id={tid} (test_key={test_key}) -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Run Influence Functions and save per-test scores")

    p.add_argument("--experiment", choices=["mnist", "multimodal"], required=True)
    p.add_argument("--ckpt_path", type=str, required=True)

    # IF params
    p.add_argument("--recursion_depth", type=int, default=1000)
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--scale", type=float, default=25.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)

    # Single-test mode
    p.add_argument("--test_id", type=int, default=0)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--save_scores_npy", type=str, default=None)
    p.add_argument("--save_topk_ids_txt", type=str, default=None)

    # All tests mode
    p.add_argument("--all_test_ids", type=int, default=0)     # 1 = run all tests
    p.add_argument("--save_all_dir", type=str, default=None)
    p.add_argument("--all_test_limit", type=int, default=0)   # 0 = all
    p.add_argument("--resume_all_test", type=int, default=1)  # 1 = skip existing

    # MNIST
    p.add_argument("--mnist_root", type=str, default="./mnist")
    p.add_argument("--mnist_train_limit", type=int, default=55000)
    p.add_argument("--mnist_test_limit", type=int, default=10000)

    # Multimodal
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--test_jsonl", type=str, default=None)
    p.add_argument("--image_root", type=str, default="./")

    # Multimodal model shape (must match train_base.py)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_image_tokens_per_patch", type=int, default=12)
    p.add_argument("--max_seq_len", type=int, default=768)

    # internal
    p.add_argument("--tmp_dir", type=str, default="./outputs/_tmp_if")

    return p


def main():
    setup_logging()
    args = build_parser().parse_args()

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt_path}")

    if args.experiment == "multimodal":
        if not args.train_jsonl or not args.test_jsonl:
            raise ValueError("multimodal requires --train_jsonl and --test_jsonl")

    device = get_device()
    logging.info(f"Device: {device}")
    logging.info(f"Experiment: {args.experiment}")
    logging.info(f"Checkpoint: {args.ckpt_path}")

    if args.all_test_ids == 1:
        run_all_tests(args, device)
    else:
        run_single_test(args, device)


if __name__ == "__main__":
    main()
