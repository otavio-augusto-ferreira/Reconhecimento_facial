"""Uma época de treino para as três perdas (ce, arcface, triplet)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from facerec.training.data import augment, pk_batches, random_batches, standardize, to_tensor
from facerec.training.losses import CosineClassifier, batch_hard_triplet_loss
from facerec.training.model import set_train_mode

LOSSES = ("ce", "arcface", "triplet")


def make_optimizer(model, head, lr_backbone: float, lr_head: float, weight_decay: float):
    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr_backbone}]
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": lr_head})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_epoch(
    model,
    head: CosineClassifier | None,
    loss_name: str,
    crops: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str,
    optimizer,
    scheduler=None,
    batch_size: int,
    p: int,
    k: int,
    steps: int | None,
    triplet_margin: float,
    use_augment: bool,
    rng: np.random.Generator,
    gen: torch.Generator,
    device,
) -> float:
    """Percorre as imagens de treino uma vez (ou `steps` lotes P x K no triplet). Devolve a perda média."""
    if loss_name not in LOSSES:
        raise ValueError(f"perda desconhecida: {loss_name!r} (use {LOSSES})")
    set_train_mode(model, mode)
    if head is not None:
        head.train()
    if loss_name == "triplet":
        batches = pk_batches(labels, p, k, steps or max(1, len(labels) // (p * k)), rng)
    else:
        batches = random_batches(len(labels), batch_size, rng)

    total, n = 0.0, 0
    for idx in batches:
        idx = np.sort(idx)  # leitura em ordem; a ordem dentro do lote não importa
        x = to_tensor(crops[idx], device)
        y = torch.from_numpy(labels[idx]).long().to(device)
        if use_augment:
            x = augment(x, gen)
        emb = model(standardize(x))
        if loss_name == "triplet":
            loss = batch_hard_triplet_loss(emb, y, triplet_margin)
        else:
            loss = F.cross_entropy(head(emb, y), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)
