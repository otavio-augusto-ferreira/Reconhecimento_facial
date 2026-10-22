"""Funções de perda para aprender embeddings de face.

- `CosineClassifier` com `margin = 0`: softmax sobre similaridades de cosseno com escala (a "CE" dos experimentos);
- `CosineClassifier` com `margin > 0`: ArcFace (margem angular aditiva na classe correta);
- `batch_hard_triplet_loss`: triplet loss com a amostra mais difícil de cada âncora no lote.
Nos três casos o embedding é comparado por cosseno, a mesma métrica usada na avaliação e na aplicação.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """Classificador por cosseno: logits = scale * cos(theta), com margem angular opcional (ArcFace)."""

    def __init__(self, dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.0):
        super().__init__()
        self.scale, self.margin, self.n_classes = scale, margin, n_classes
        self.weight = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.xavier_uniform_(self.weight)

    @torch.no_grad()
    def init_from_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Cada classe começa na média (normalizada) dos embeddings das suas imagens: o treino parte de um
        classificador já razoável e não destrói o que o modelo pré-treinado aprendeu."""
        device = self.weight.device  # os embeddings podem vir da CPU enquanto o classificador está na GPU
        emb = F.normalize(embeddings.float().to(device), dim=1)
        proto = torch.zeros_like(self.weight)
        proto.index_add_(0, labels.long().to(device), emb)
        present = proto.norm(dim=1) > 0
        proto[present] = F.normalize(proto[present], dim=1)
        proto[~present] = self.weight[~present]
        self.weight.copy_(proto)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cos = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is not None and self.margin > 0:
            m = self.margin
            phi = torch.cos(torch.acos(cos) + m)
            # quando theta + m passa de pi, cos(theta + m) deixa de ser monótono: usa-se uma aproximação linear
            phi = torch.where(cos > math.cos(math.pi - m), phi, cos - math.sin(math.pi - m) * m)
            target = F.one_hot(labels, self.n_classes).bool()
            cos = torch.where(target, phi, cos)
        return cos * self.scale


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Para cada âncora: (positivo mais distante) - (negativo mais próximo) + margem, truncado em 0.

    Distância euclidiana entre embeddings normalizados. Âncoras sem positivo ou sem negativo no lote são ignoradas.
    """
    emb = F.normalize(embeddings, dim=1)
    dist = torch.sqrt((2.0 - 2.0 * emb @ emb.T).clamp_min(1e-12))  # clamp evita gradiente infinito em d = 0
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos, neg = same & ~eye, ~same
    hardest_pos = dist.masked_fill(~pos, 0.0).max(dim=1).values
    hardest_neg = dist.masked_fill(~neg, float("inf")).min(dim=1).values
    valid = pos.any(dim=1) & neg.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    return F.relu(hardest_pos - hardest_neg + margin)[valid].mean()
