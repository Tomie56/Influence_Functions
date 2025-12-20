import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import build_model, SimpleTokenizer
from data_utils_mnist import build_dataloader_mnist
from data_utils import build_dataloader
from influence_function import calc_influence_single

def parse_args():
    parser = argparse.ArgumentParser("Calculate Influence Functions (IF) for single/all test samples")
    # Experiment config
    parser.add_argument("--experiment", choices=["mnist", "multimodal"], required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    # Data config
    parser.add_argument("--mnist_root", type=str, default="./mnist")
    parser.add_argument("--mnist_train_limit", type=int, default=55000)
    parser.add_argument("--mnist_test_limit", type=int, default=10000)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--test_jsonl", type=str, default=None)
    parser.add_argument("--image_root", type=str, default="./")
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=768)
    # IF config
    parser.add_argument("--test_id", type=int, default=0)
    parser.add_argument("--all_test_ids", action="store_true", help="Calculate IF for all test samples")
    parser.add_argument("--recursion_depth", type=int, default=1000)
    parser.add_argument("--r_averaging", type=int, default=10)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=25.0)
    parser.add_argument("--loss_func", type=str, default="cross_entropy")
    # Output config
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, -1 for CPU")
    return parser.parse_args()

def load_model_and_tokenizer(args, device):
    """Load model and tokenizer (handle both single state dict and wrapped dict with extra params)"""
    # Load model based on experiment type
    if args.experiment == "mnist":
        model = build_model(
            experiment="mnist",
            weight_decay=0.01, 
            is_multi=True
        ).to(device)
        
        # Load checkpoint and extract model state dict
        ckpt = torch.load(args.ckpt_path, map_location=device)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                model_state_dict = ckpt["model_state_dict"]
            else:
                model_state_dict = ckpt
        else:
            raise TypeError(f"Invalid checkpoint type: {type(ckpt)}")
        
        # Load model state dict
        model.load_state_dict(model_state_dict)
        return model, None
    else:
        # Load multimodal model and tokenizer
        ckpt = torch.load(args.ckpt_path, map_location=device)
        if "tokenizer_config" in ckpt:
            tokenizer_config = ckpt["tokenizer_config"]
        else:
            tokenizer_config = {
                "vocab_size": 10000,
                "max_seq_len": args.max_seq_len
            }
        
        tokenizer = SimpleTokenizer(
            vocab_size=tokenizer_config["vocab_size"],
            max_seq_len=tokenizer_config["max_seq_len"]
        )
        
        # Extract multimodal model state dict
        model = build_model(
            experiment="multimodal",
            vocab_size=tokenizer_config["vocab_size"],
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=tokenizer_config["max_seq_len"]
        ).to(device)
        
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        
        return model, tokenizer

def build_data_loaders(args, tokenizer, device):
    """Build train and test dataloaders"""
    # Build train and test dataloaders
    if args.experiment == "mnist":
        train_loader, _ = build_dataloader_mnist(
            data_root=args.mnist_root,
            train=True,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            limit_size=args.mnist_train_limit,
            return_dataset=False
        )
        test_loader, test_dataset = build_dataloader_mnist(
            data_root=args.mnist_root,
            train=False,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            limit_size=args.mnist_test_limit,
            return_dataset=True
        )
    else:
        train_loader, _ = build_dataloader(
            data_path=args.train_jsonl,
            tokenizer=tokenizer,
            image_root=args.image_root,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            return_dataset=False
        )
        test_loader, test_dataset = build_dataloader(
            data_path=args.test_jsonl,
            tokenizer=tokenizer,
            image_root=args.image_root,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            return_dataset=True
        )
    return train_loader, test_loader, test_dataset

def get_train_ids(train_loader):
    """Extract stable training sample IDs"""
    train_ids = []
    for batch in train_loader:
        if isinstance(batch, dict):
            train_ids.extend(batch["idx"].cpu().numpy().tolist())
        else:
            _, _, batch_ids = batch
            train_ids.extend(batch_ids.cpu().numpy().tolist())
    return np.array(train_ids)

def save_if_results(args, test_id, influences, train_ids):
    """Save IF results and top-K IDs"""
    def convert_to_numpy(elem):
        if isinstance(elem, torch.Tensor):
            return elem.cpu().numpy()
        return elem 

    # Create save directory if not exists
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    tmp_dir = save_dir / "tmp_if"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    # Save IF scores and train IDs
    influence_path = save_dir / f"influences_test_{test_id}.npz"
    np.savez(
        influence_path,
        scores=np.array([convert_to_numpy(inf) for inf in influences]),
        train_ids=train_ids
    )
    print(f"[Test {test_id}] IF results saved to {influence_path}")

    # Select and save Top-K train IDs (sorted by absolute influence)
    abs_influences = np.abs([convert_to_numpy(inf) for inf in influences])
    topk_indices = np.argsort(abs_influences)[-args.topk:]
    topk_train_ids = train_ids[topk_indices]
    
    topk_path = save_dir / f"topk_ids_test_{test_id}.txt"
    with open(topk_path, "w") as f:
        for tid in topk_train_ids:
            f.write(f"{tid}\n")
    print(f"[Test {test_id}] Top-{args.topk} train IDs saved to {topk_path}")

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and data
    model, tokenizer = load_model_and_tokenizer(args, device)
    model.eval()
    train_loader, test_loader, test_dataset = build_data_loaders(args, tokenizer, device)
    train_ids = get_train_ids(train_loader)

    # Determine test IDs to process
    test_ids = [args.test_id]
    if args.all_test_ids:
        test_ids = list(range(len(test_dataset)))
        print(f"Will calculate IF for {len(test_ids)} test samples")

    # Calculate IF for each test sample
    for test_id in tqdm(test_ids, desc="Processing test samples"):
        influences, harmful, helpful, _ = calc_influence_single(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            test_id_num=test_id,
            recursion_depth=args.recursion_depth,
            r=args.r_averaging,
            gpu=args.gpu,
            damp=args.damp,
            scale=args.scale,
            loss_func=args.loss_func
        )
        # Save results
        save_if_results(args, test_id, influences, train_ids)

    print("IF calculation and saving completed!")

if __name__ == "__main__":
    main()