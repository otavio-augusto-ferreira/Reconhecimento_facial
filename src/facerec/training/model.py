"""Carrega o FaceNet e controla quais camadas treinam."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from facerec.training.data import standardize, to_tensor

# Blocos do InceptionResnetV1 (facenet-pytorch) que treinam em cada modo. None = a rede toda.
TRAINABLE_MODES: dict[str, tuple[str, ...] | None] = {
    "last_block": ("repeat_3", "block8", "last_linear", "last_bn"),  # último estágio Inception-ResNet + projeção final
    "full": None,
}


def load_facenet(pretrained: str | None = "vggface2", device: str = "cpu", dropout: float = 0.0) -> nn.Module:
    """InceptionResnetV1 sem a camada de classificação. `pretrained=None` dá pesos aleatórios (testes)."""
    from facenet_pytorch import InceptionResnetV1

    return InceptionResnetV1(pretrained=pretrained, classify=False, dropout_prob=dropout).to(device)


def _trainable_modules(model: nn.Module, mode: str) -> list[nn.Module]:
    names = TRAINABLE_MODES[mode]
    return [model] if names is None else [getattr(model, n) for n in names]


def set_trainable(model: nn.Module, mode: str) -> int:
    """Congela tudo e libera os parâmetros do modo. Devolve o número de parâmetros treináveis."""
    if mode not in TRAINABLE_MODES:
        raise ValueError(f"modo desconhecido: {mode!r} (use {sorted(TRAINABLE_MODES)})")
    for p in model.parameters():
        p.requires_grad = False
    for module in _trainable_modules(model, mode):
        for p in module.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_train_mode(model: nn.Module, mode: str) -> None:
    """Só os blocos treináveis ficam em train(); o resto fica em eval(), para a normalização em lote
    (BatchNorm) das camadas congeladas não mudar as estatísticas que o modelo pré-treinado aprendeu."""
    model.eval()
    for module in _trainable_modules(model, mode):
        module.train()


@torch.no_grad()
def embed_crops(model: nn.Module, crops: np.ndarray, device: str | torch.device, batch_size: int = 128) -> np.ndarray:
    """Embeddings L2-normalizados (n, 512) de recortes uint8 (n, 160, 160, 3), com o modelo em eval()."""
    model.eval()
    out = []
    for i in range(0, len(crops), batch_size):
        x = standardize(to_tensor(crops[i : i + batch_size], device))
        out.append(F.normalize(model(x), p=2, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 512), np.float32)
