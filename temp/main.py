import os
import logging
import argparse
import torch

from influence_function import calculate_influences
from influence_loo import train_base_model, calculate_loo_loss_changes, load_loo_loss_changes
from vis import plot_influence_distribution, plot_influence_vs_leave_one_out


def setup_logging(output_dir: str, log_file: str | None):
    os.makedirs(output_dir, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        handlers=handlers,
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_mnist(args, device: torch.device):
    from model import TorchLogisticRegression, MNIST_INPUT_DIM, MNIST_NUM_CLASSES
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

    model_init_kwargs = dict(
        input_dim=MNIST_INPUT_DIM,
        num_classes=MNIST_NUM_CLASSES,
        loss_reduction="mean",
        l2_reg=args.lr_l2_reg,  # 若你用 weight_decay 做 L2，把它设为0即可
    )

    model = TorchLogisticRegression(**model_init_kwargs).to(device)

    if args.skip_base_train:
        if not args.model_ckpt_path or not os.path.exists(args.model_ckpt_path):
            raise ValueError("--skip_base_train requires an existing --model_ckpt_path")
        model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device))
        logging.info(f"Loaded base model: {args.model_ckpt_path}")
    else:
        model = train_base_model(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.base_train_epochs,
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer,
            lbfgs_max_iter=args.lbfgs_max_iter,
        )
        ckpt_path = args.model_ckpt_path or os.path.join(args.output_dir, "mnist_lr_base.pth")
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"Saved base model: {ckpt_path}")

    influences, harmful, helpful = calculate_influences(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_test_samples=args.num_test_samples,
        recursion_depth=args.recursion_depth,
        damp=args.damp,
        scale=args.scale,
        save_intermediate=True,
        output_dir=args.output_dir,
        top_k=args.top_k_report,
    )

    if not influences:
        logging.warning("No influence results, stop.")
        return

    test_id = next(iter(influences.keys()))

    if args.skip_loo_calc:
        loo = load_loo_loss_changes(args.loo_save_path, test_id)
    else:
        loo = calculate_loo_loss_changes(
            base_model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            loo_train_epochs=args.loo_train_epochs,
            loo_lr=args.loo_lr,
            max_loo_samples=args.max_loo_samples,
            test_idx=test_id,
            loo_save_path=args.loo_save_path,
            model_class=TorchLogisticRegression,
            model_init_kwargs=model_init_kwargs,
            mnist_root=args.mnist_root,
            build_dataloader_mnist_fn=build_dataloader_mnist,
            optimizer_type=args.optimizer,
            weight_decay=args.weight_decay,
            lbfgs_max_iter=args.lbfgs_max_iter,
        )

    plot_influence_distribution(influences=influences, save_dir=args.output_dir, test_idx=test_id, top_n=args.top_k_report)
    predicted = influences[test_id][: len(loo)]
    plot_influence_vs_leave_one_out(
        actual_loss_changes=loo,
        predicted_influences=predicted,
        save_dir=args.output_dir,
        test_idx=test_id,
        top_k=args.top_k_figure2,
        model_type=args.model_type_figure2,
    )

    logging.info("MNIST experiment done.")


def run_multimodal(args, device: torch.device):
    from model import MiniMultimodalModel, SimpleTokenizer, DEFAULT_VOCAB_SIZE, DEFAULT_IMAGE_EMBED_DIM
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
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        image_root=args.image_root,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        return_dataset=True,
        **dataset_kwargs,
    )
    test_loader, test_dataset = build_dataloader(
        data_path=args.test_data_path,
        tokenizer=tokenizer,
        image_root=args.image_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        return_dataset=True,
        **dataset_kwargs,
    )

    model_init_kwargs = dict(
        vocab_size=DEFAULT_VOCAB_SIZE,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        image_embed_dim=DEFAULT_IMAGE_EMBED_DIM,
        image_size=args.image_size,
        num_image_tokens_per_patch=args.num_image_tokens_per_patch,
        max_seq_len=args.max_seq_len,
    )

    model = MiniMultimodalModel(**model_init_kwargs).to(device)

    if args.skip_base_train:
        if not args.model_ckpt_path or not os.path.exists(args.model_ckpt_path):
            raise ValueError("--skip_base_train requires an existing --model_ckpt_path")
        model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device))
        logging.info(f"Loaded base model: {args.model_ckpt_path}")
    else:
        model = train_base_model(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.base_train_epochs,
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            optimizer_type="adamw",
        )
        ckpt_path = args.model_ckpt_path or os.path.join(args.output_dir, "multimodal_base.pth")
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"Saved base model: {ckpt_path}")

    influences, harmful, helpful = calculate_influences(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_test_samples=args.num_test_samples,
        recursion_depth=args.recursion_depth,
        damp=args.damp,
        scale=args.scale,
        save_intermediate=True,
        output_dir=args.output_dir,
        top_k=args.top_k_report,
    )

    if not influences:
        logging.warning("No influence results, stop.")
        return

    test_id = next(iter(influences.keys()))

    if args.skip_loo_calc:
        loo = load_loo_loss_changes(args.loo_save_path, test_id)
    else:
        loo = calculate_loo_loss_changes(
            base_model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            tokenizer=tokenizer,
            image_root=args.image_root,
            dataset_kwargs=dataset_kwargs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            loo_train_epochs=args.loo_train_epochs,
            loo_lr=args.loo_lr,
            max_loo_samples=args.max_loo_samples,
            test_idx=test_id,
            loo_save_path=args.loo_save_path,
            model_class=MiniMultimodalModel,
            model_init_kwargs=model_init_kwargs,
            optimizer_type="adamw",
            weight_decay=args.weight_decay,
        )

    plot_influence_distribution(influences=influences, save_dir=args.output_dir, test_idx=test_id, top_n=args.top_k_report)
    predicted = influences[test_id][: len(loo)]
    plot_figure2_middle(
        actual_loss_changes=loo,
        predicted_influences=predicted,
        save_dir=args.output_dir,
        test_idx=test_id,
        top_k=args.top_k_figure2,
        model_type=args.model_type_figure2,
    )

    logging.info("Multimodal experiment done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Influence Function + LOO runner")

    p.add_argument("--experiment", choices=["mnist", "multimodal"], required=True)
    p.add_argument("--output_dir", default="./influence_results")
    p.add_argument("--log_file", default=None)
    p.add_argument("--model_ckpt_path", default=None)
    p.add_argument("--skip_base_train", action="store_true")
    p.add_argument("--skip_loo_calc", action="store_true")
    p.add_argument("--loo_save_path", default="./loo_loss_changes.npy")

    p.add_argument("--base_train_epochs", type=int, default=5)
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--num_test_samples", type=int, default=1)
    p.add_argument("--recursion_depth", type=int, default=1000)
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--scale", type=float, default=25.0)

    p.add_argument("--loo_train_epochs", type=int, default=3)
    p.add_argument("--loo_lr", type=float, default=1e-3)
    p.add_argument("--max_loo_samples", type=int, default=100)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--top_k_report", type=int, default=50)
    p.add_argument("--top_k_figure2", type=int, default=500)
    p.add_argument("--model_type_figure2", default="approx")

    # MNIST
    p.add_argument("--mnist_root", default="./mnist")
    p.add_argument("--mnist_train_limit", type=int, default=None)  # 论文可设 55000
    p.add_argument("--mnist_test_limit", type=int, default=None)
    p.add_argument("--optimizer", choices=["adamw", "lbfgs"], default="lbfgs")
    p.add_argument("--lbfgs_max_iter", type=int, default=20)
    p.add_argument("--lr_l2_reg", type=float, default=0.0)  # 若不用 weight_decay 做 L2，可设 0.01

    # Multimodal
    p.add_argument("--train_data_path", default=None)
    p.add_argument("--test_data_path", default=None)
    p.add_argument("--image_root", default="./")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_image_tokens_per_patch", type=int, default=12)
    p.add_argument("--max_seq_len", type=int, default=768)

    return p


def main():
    args = build_parser().parse_args()
    setup_logging(args.output_dir, args.log_file)

    device = get_device()
    logging.info(f"Device: {device}")

    if args.experiment == "mnist":
        run_mnist(args, device)
    else:
        if not args.train_data_path or not args.test_data_path:
            raise ValueError("multimodal requires --train_data_path and --test_data_path")
        run_multimodal(args, device)

    logging.info("Done.")


if __name__ == "__main__":
    main()
