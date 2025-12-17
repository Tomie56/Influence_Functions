import os
import logging
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext


def _amp_ctx(device: torch.device):
    """Use bfloat16 autocast on CUDA; no-op elsewhere."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_trainable_params(model: nn.Module) -> List[torch.Tensor]:
    """Return trainable parameters (requires_grad=True)."""
    return [p for p in model.parameters() if p.requires_grad]


def hvp(loss: torch.Tensor, params: List[torch.Tensor], vec: List[torch.Tensor]) -> List[torch.Tensor]:
    """Hessian-vector product for the given loss and parameter list."""
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    hv = torch.autograd.grad(grads, params, grad_outputs=vec, retain_graph=False, create_graph=False)
    return list(hv)


def _normalize_batch(batch: Any) -> Dict[str, Any]:
    """
    Normalize dataloader output to a dict.

    Supported:
      - multimodal: dict (kept as-is)
      - torchvision-style MNIST: (x, y) or (x, y, idx)
    """
    if isinstance(batch, dict):
        return batch

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            x, y = batch
            return {"x": x, "labels": y}
        if len(batch) == 3:
            x, y, idx = batch
            return {"x": x, "labels": y, "idx": idx}
        raise TypeError(f"Unsupported tuple/list batch length={len(batch)}")

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor fields to device; keep non-tensors unchanged."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _is_mnist_batch(batch: Dict[str, torch.Tensor]) -> bool:
    return ("x" in batch) and ("labels" in batch)


def _is_multimodal_batch(batch: Dict[str, torch.Tensor]) -> bool:
    return ("input_ids" in batch) and ("labels" in batch)


def _get_sample_id(batch: Dict[str, torch.Tensor], i: int) -> int:
    """
    Return a stable sample id when available.

    - multimodal: sample_index[i]
    - mnist: idx[i] (if provided by dataset/collate)
    - fallback: i (batch-local index)
    """
    if "sample_index" in batch:
        return int(batch["sample_index"][i].item())
    if "idx" in batch:
        return int(batch["idx"][i].item())
    return int(i)


def _slice_single(batch: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Slice a batch dict into a single-item batch (keeps batch dimension)."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v[i : i + 1] if torch.is_tensor(v) else v
    return out


def _forward_loss(model: nn.Module, batch: Any, device: torch.device) -> torch.Tensor:
    """Compute per-batch loss for MNIST logistic regression or the multimodal model."""
    batch = _to_device_batch(_normalize_batch(batch), device)

    with _amp_ctx(device):
        if _is_multimodal_batch(batch):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                image_flags=batch.get("image_flags"),
                labels=batch.get("labels"),
            )
            loss = out.loss
        elif _is_mnist_batch(batch):
            out = model(x=batch["x"], labels=batch["labels"])
            loss = out.loss
        else:
            raise KeyError(
                f"Unknown batch format. Keys={list(batch.keys())}. "
                "Expect multimodal(input_ids/labels/...) or mnist(x/labels[/idx])."
            )

    if loss is None:
        raise RuntimeError("Model returned loss=None. Ensure labels are provided and loss is computed.")
    return loss


def grad_z(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    create_graph: bool = False,
) -> List[torch.Tensor]:
    """Gradient of loss w.r.t. trainable params for one batch."""
    model.zero_grad(set_to_none=True)
    loss = _forward_loss(model, batch, device=device)

    params = getattr(model, "_if_trainable_params", None) or get_trainable_params(model)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=create_graph)
    return [torch.nan_to_num(g, 0.0, 0.0, 0.0) for g in grads]


def s_test(
    model: nn.Module,
    test_batch: Any,
    train_loader: DataLoader,
    device: torch.device,
    recursion_depth: int = 200,
    damp: float = 0.01,
    scale: float = 25.0,
) -> List[torch.Tensor]:
    """
    Approximate H^{-1} v using the LiSSA recursion.
    v = grad(loss_test)
    """
    params = getattr(model, "_if_trainable_params", None) or get_trainable_params(model)
    if len(params) == 0:
        raise RuntimeError("No trainable params")

    v = grad_z(model, test_batch, device=device, create_graph=False)
    h = [gi.detach().clone() for gi in v]

    it = iter(train_loader)
    for step in range(recursion_depth):
        try:
            train_batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            train_batch = next(it)

        model.zero_grad(set_to_none=True)
        loss = _forward_loss(model, train_batch, device=device)

        hv = hvp(loss, params, h)
        h = [v_i + (1.0 - damp) * h_i - hv_i / scale for v_i, h_i, hv_i in zip(v, h, hv)]

        if (step % 50 == 0) or (step == recursion_depth - 1):
            logging.info(f"[s_test] {step+1}/{recursion_depth}")

    return h


def calculate_influences(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_test_samples: int = 1,
    recursion_depth: int = 200,
    damp: float = 0.01,
    scale: float = 25.0,
    save_intermediate: bool = True,
    output_dir: str = "./influence_results",
    top_k: int = 50,
) -> Tuple[Dict[int, List[float]], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Compute influence scores for up to num_test_samples test points.

    Saved per test_id:
      influences_test_{test_id}.npz with:
        - scores: float32 [num_train]
        - train_ids: int64 [num_train] (sorted ascending)

    Returns:
      influences[test_id] -> scores aligned with sorted train_ids
      helpful[test_id]   -> train_ids with most negative scores (top_k)
      harmful[test_id]   -> train_ids with most positive scores (top_k)
    """
    os.makedirs(output_dir, exist_ok=True)
    interm = os.path.join(output_dir, "intermediate")
    if save_intermediate:
        os.makedirs(interm, exist_ok=True)

    params = get_trainable_params(model)
    if len(params) == 0:
        raise RuntimeError("No trainable params")
    model._if_trainable_params = params

    influences: Dict[int, List[float]] = {}
    harmful: Dict[int, List[int]] = {}
    helpful: Dict[int, List[int]] = {}

    processed = 0
    for raw_test_batch in tqdm(test_loader, desc="test"):
        if processed >= num_test_samples:
            break

        test_batch = _normalize_batch(raw_test_batch)
        any_key = next(k for k, v in test_batch.items() if torch.is_tensor(v))
        B = int(test_batch[any_key].size(0))

        for bi in range(B):
            if processed >= num_test_samples:
                break

            test_id = _get_sample_id(test_batch, bi)
            single_test = _slice_single(test_batch, bi)
            logging.info(f"== Test {test_id} ({processed+1}/{num_test_samples}) ==")

            svec = s_test(
                model=model,
                test_batch=single_test,
                train_loader=train_loader,
                device=device,
                recursion_depth=recursion_depth,
                damp=damp,
                scale=scale,
            )

            if save_intermediate:
                torch.save([t.detach().cpu() for t in svec], os.path.join(interm, f"s_test_{test_id}.pt"))

            scores_by_id: Dict[int, float] = {}
            gz_dir = os.path.join(interm, f"grad_z_test_{test_id}")
            if save_intermediate:
                os.makedirs(gz_dir, exist_ok=True)

            for raw_train_batch in tqdm(train_loader, desc=f"train (test={test_id})", leave=False):
                train_batch = _normalize_batch(raw_train_batch)
                any_key_tr = next(k for k, v in train_batch.items() if torch.is_tensor(v))
                BB = int(train_batch[any_key_tr].size(0))

                for bj in range(BB):
                    train_id = _get_sample_id(train_batch, bj)
                    single_train = _slice_single(train_batch, bj)

                    gz = grad_z(model, single_train, device=device, create_graph=False)

                    if save_intermediate:
                        torch.save([g.detach().cpu() for g in gz], os.path.join(gz_dir, f"grad_z_{train_id}.pt"))

                    score = -sum((g * s).sum() for g, s in zip(gz, svec))
                    scores_by_id[train_id] = float(score.item())

            train_ids_sorted = sorted(scores_by_id.keys())
            scores = np.array([scores_by_id[i] for i in train_ids_sorted], dtype=np.float32)

            influences[test_id] = scores.tolist()

            if top_k > 0:
                order = np.argsort(scores)
                helpful[test_id] = [train_ids_sorted[i] for i in order[:top_k].tolist()]
                harmful[test_id] = [train_ids_sorted[i] for i in order[-top_k:].tolist()]
            else:
                helpful[test_id] = []
                harmful[test_id] = []

            np.savez(
                os.path.join(output_dir, f"influences_test_{test_id}.npz"),
                scores=scores,
                train_ids=np.array(train_ids_sorted, dtype=np.int64),
            )

            processed += 1

    np.save(os.path.join(output_dir, "influences.npy"), influences, allow_pickle=True)
    np.save(os.path.join(output_dir, "helpful_samples.npy"), helpful, allow_pickle=True)
    np.save(os.path.join(output_dir, "harmful_samples.npy"), harmful, allow_pickle=True)

    return influences, harmful, helpful
