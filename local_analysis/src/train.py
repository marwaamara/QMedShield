"""
qmed.train — training and evaluation primitives.

Model selection is done on validation data only, never on the test set.

`train_model` accepts a validation loader only, and `assert_no_leakage`
(validation gate G3) refuses to run if test indices appear in the training or
validation folds. Test evaluation happens exactly once, in the experiment
script, after training returns.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import CFG, epochs_for, lr_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Validation gate G3 -- leakage assertion
# ---------------------------------------------------------------------------
def assert_no_leakage(train_idx: np.ndarray, val_idx: np.ndarray,
                      test_idx: np.ndarray) -> None:
    """Raise if the test indices intersect training or validation indices.

    Assert it; don't trust it.
    """
    tr, va, te = set(map(int, train_idx)), set(map(int, val_idx)), set(map(int, test_idx))
    if tr & te:
        raise AssertionError(
            f"LEAKAGE (gate G3): {len(tr & te)} test indices appear in the "
            f"training split."
        )
    if va & te:
        raise AssertionError(
            f"LEAKAGE (gate G3): {len(va & te)} test indices appear in the "
            f"validation split."
        )
    if tr & va:
        raise AssertionError(
            f"LEAKAGE (gate G3): {len(tr & va)} indices are shared between the "
            f"training and validation splits."
        )


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------
def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = None,
                shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size or CFG.train.BATCH_SIZE,
                      shuffle=shuffle, num_workers=0)


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """sqrt-dampened inverse frequency, as in the original work."""
    counts = np.bincount(y, minlength=n_classes)
    w = np.array([1.0 / np.sqrt(max(c, 1)) for c in counts], dtype=np.float32)
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, n_classes: int = 2, device=DEVICE) -> Dict:
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
        preds_all.append(logits.argmax(1).cpu().numpy())
        labels_all.append(yb.numpy())

    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)

    average = "binary" if n_classes == 2 else "weighted"
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0
    )
    try:
        auc = (roc_auc_score(labels, probs[:, 1]) if n_classes == 2
               else roc_auc_score(labels, probs, multi_class="ovr", average="weighted"))
    except Exception:
        auc = float("nan")

    # Per-class metrics -- with the enlarged test set these are
    # finally reportable for every class, including the rare ones.
    pc_p, pc_r, pc_f1, pc_sup = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(n_classes)), zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc),
        "per_class": {
            "precision": pc_p.tolist(), "recall": pc_r.tolist(),
            "f1": pc_f1.tolist(), "support": pc_sup.tolist(),
        },
        "confusion_matrix": confusion_matrix(
            labels, preds, labels=list(range(n_classes))
        ).tolist(),
        "_predictions": preds, "_labels": labels, "_probabilities": probs,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device=DEVICE):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG.train.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, task: str,
                class_weights: Optional[torch.Tensor] = None,
                smoke: bool = False, checkpoint_path: Optional[str] = None,
                device=DEVICE, verbose: bool = True,
                epochs_override: Optional[int] = None) -> Dict:
    """Train with early stopping and val-only checkpoint selection.

    NOTE the signature: there is no `test_loader` parameter, and there is no
    way to pass one. That is deliberate -- it makes test-set selection unexpressible.

    Epochs and learning rates come from qmed.config via `epochs_for(task)` and
    `lr_for(task)`; no caller may override them.
    """
    # `epochs_override` exists for ONE purpose: the qubit/depth sweep, where
    # every configuration must get the SAME reduced budget so the comparison is
    # not confounded by letting each run to its own convergence. It is keyword,
    # explicit, printed, and written into the history -- so a schedule can never
    # drift silently. The ablation and the headline
    # never pass it.
    epochs = epochs_override or epochs_for(task, smoke)
    if epochs_override and verbose:
        print(f"  [budget] epochs_override={epochs_override} "
              f"(schedule for '{task}' is {epochs_for(task, smoke)})")
    lr, lr_min = lr_for(task)
    n_classes = 2 if task == "binary" else 8

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=CFG.train.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "val_auc": []}
    best_f1, best_state, since_best = -1.0, None, 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        vm = evaluate(model, val_loader, n_classes, device)   # VALIDATION only
        scheduler.step()

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_acc"].append(vm["accuracy"])
        history["val_f1"].append(vm["f1"])
        history["val_auc"].append(vm["auc"])

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"  epoch {epoch:3d}/{epochs}  train_acc={ta:.4f}  "
                  f"val_f1={vm['f1']:.4f}  val_auc={vm['auc']:.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        if checkpoint_path and CFG.train.CHECKPOINT_EVERY_N_EPOCHS > 0 \
                and epoch % CFG.train.CHECKPOINT_EVERY_N_EPOCHS == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "best_f1": best_f1, "best_state": best_state},
                       checkpoint_path)

        if not smoke and epoch >= CFG.train.EARLY_STOP_MIN_EPOCHS \
                and since_best >= CFG.train.EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val_f1={best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["epochs_run"] = epoch
    history["epochs_budget"] = epochs
    history["epochs_override"] = epochs_override
    history["best_val_f1"] = best_f1
    history["train_wall_s"] = time.time() - t_start
    return history
