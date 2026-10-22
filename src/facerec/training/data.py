"""Amostragem de lotes e aumento de dados para o fine-tuning."""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch

from facerec.data.splits import Item


def limit_per_identity(items: list[Item], max_images: int | None, seed: int = 42) -> list[Item]:
    """No máximo `max_images` por identidade (amostragem determinística), para que pessoas muito fotografadas
    não dominem o treino."""
    if max_images is None:
        return list(items)
    groups: dict[str, list[str]] = defaultdict(list)
    for path, identity in items:
        groups[identity].append(path)
    rng = random.Random(seed)
    out: list[Item] = []
    for identity in sorted(groups):
        paths = sorted(groups[identity])
        if len(paths) > max_images:
            paths = sorted(rng.sample(paths, max_images))
        out.extend((p, identity) for p in paths)
    return out


def random_batches(n: int, batch_size: int, rng: np.random.Generator):
    """Índices embaralhados em lotes completos (o último lote incompleto é descartado)."""
    order = rng.permutation(n)
    for i in range(0, n - batch_size + 1, batch_size):
        yield order[i : i + batch_size]


def pk_batches(labels: np.ndarray, p: int, k: int, steps: int, rng: np.random.Generator):
    """Lotes P x K para o triplet loss: P identidades com K imagens cada (com reposição se faltarem imagens)."""
    by_label: dict[int, np.ndarray] = {}
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) >= 2:
            by_label[int(lab)] = idx
    classes = np.array(sorted(by_label))
    if len(classes) < 2:
        raise ValueError("o triplet loss precisa de pelo menos 2 identidades com >= 2 imagens")
    for _ in range(steps):
        chosen = rng.choice(classes, size=min(p, len(classes)), replace=False)
        batch = [rng.choice(by_label[int(c)], size=k, replace=len(by_label[int(c)]) < k) for c in chosen]
        yield np.concatenate(batch)


def to_tensor(crops: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """uint8 (n, H, W, 3) -> float (n, 3, H, W) em [0, 255]."""
    return torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float()


def augment(x: torch.Tensor, gen: torch.Generator, max_shift: int = 8) -> torch.Tensor:
    """Aumento leve, em [0, 255]: espelhamento, brilho, contraste e pequena translação.

    Não usa rotação nem recortes grandes: o recorte já vem alinhado pelo detector.
    """
    n, _, h, w = x.shape
    dev = x.device

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device="cpu").to(dev)

    flip = rand(n) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    brightness = (0.8 + 0.4 * rand(n))[:, None, None, None]
    contrast = (0.8 + 0.4 * rand(n))[:, None, None, None]
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * brightness
    if max_shift > 0:
        padded = torch.nn.functional.pad(x, (max_shift,) * 4, mode="replicate")
        shifts = torch.randint(0, 2 * max_shift + 1, (n, 2), generator=gen, device="cpu").tolist()
        x = torch.stack([padded[i, :, dy : dy + h, dx : dx + w] for i, (dy, dx) in enumerate(shifts)])
    return x.clamp(0.0, 255.0)


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Mesma padronização do FaceNetEmbedder: (x - 127.5) / 128."""
    return (x - 127.5) / 128.0
