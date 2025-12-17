import os
import json
import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np

from vis import plot_influence_distribution, plot_influence_vs_leave_one_out


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")


def read_topk_ids(path: str) -> List[int]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(int(s))
    return ids


def read_loo_jsonl(path: str) -> Tuple[float, Dict[int, float]]:
    base_loss = None
    losses: Dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            rid = int(j["id"])
            loss = float(j["loss"])
            if rid == -1:
                base_loss = loss
            else:
                losses[rid] = loss
    if base_loss is None:
        raise RuntimeError(f"Base loss (id=-1) not found in {path}")
    return base_loss, losses


def main():
    setup_logging()
    ap = argparse.ArgumentParser("Plot IF distribution + IF vs LOO after LOO stage")
    ap.add_argument("--test_id", type=int, required=True)
    ap.add_argument("--if_scores_npy", type=str, required=True)
    ap.add_argument("--topk_ids_txt", type=str, required=True)
    ap.add_argument("--loo_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--top_k_scatter", type=int, default=500)
    ap.add_argument("--model_type", type=str, default="Approx")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if_scores = np.load(args.if_scores_npy).astype(np.float32)
    topk_ids = read_topk_ids(args.topk_ids_txt)
    base_loss, loo_losses = read_loo_jsonl(args.loo_jsonl)

    influences = {args.test_id: if_scores.tolist()}
    plot_influence_distribution(influences=influences, save_dir=args.out_dir, test_idx=args.test_id, top_n=50)

    actual_changes = []
    predicted_if = []
    missing = 0

    for tid in topk_ids:
        if tid not in loo_losses:
            missing += 1
            continue
        loo_loss = loo_losses[tid]
        actual_changes.append(loo_loss - base_loss)
        predicted_if.append(float(if_scores[tid]))

    if len(actual_changes) == 0:
        raise RuntimeError("No matched LOO records found for topk_ids. Did LOO run produce entries?")

    if missing > 0:
        logging.info(f"Matched {len(actual_changes)} items; missing {missing} LOO entries (resume in progress?)")

    plot_influence_vs_leave_one_out(
        actual_loss_changes=actual_changes,
        predicted_influences=predicted_if,
        save_dir=args.out_dir,
        test_idx=args.test_id,
        top_k=args.top_k_scatter,
        model_type=args.model_type,
    )

    logging.info("Plots done.")


if __name__ == "__main__":
    main()
