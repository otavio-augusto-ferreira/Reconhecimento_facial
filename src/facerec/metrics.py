"""Métricas de verificação facial.

Convenções (iguais em todo o projeto):
- Um par tem rótulo 1 (mesma pessoa) ou 0 (pessoas diferentes).
- O score é uma similaridade: quanto maior, mais parecidas as faces.
- Um par é *aceito* quando score >= threshold.
- FAR (False Acceptance Rate) = fração dos pares NEGATIVOS aceitos.
- FRR (False Rejection Rate)  = fração dos pares POSITIVOS rejeitados.
- EER (Equal Error Rate)      = ponto em que FAR = FRR.
- Todas as funções recebem os rótulos `y` antes dos scores `s`.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- scores entre embeddings
def cosine_scores(emb: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similaridade de cosseno entre emb[a[k]] e emb[b[k]] (não depende da norma dos vetores)."""
    emb = np.asarray(emb, dtype=np.float64)
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return np.sum(unit[np.asarray(a)] * unit[np.asarray(b)], axis=1)


def euclidean_scores(emb: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Menos a distância euclidiana entre emb[a[k]] e emb[b[k]] (maior = mais parecido)."""
    emb = np.asarray(emb, dtype=np.float64)
    return -np.linalg.norm(emb[np.asarray(a)] - emb[np.asarray(b)], axis=1)


# ---------------------------------------------------------------- FAR, FRR, EER
def _split(y, s) -> tuple[np.ndarray, np.ndarray]:
    """Scores dos pares positivos e negativos, ordenados. Exige as duas classes."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("rótulos e scores devem ter o mesmo tamanho")
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("são necessários pares positivos e negativos")
    return pos, neg


def far_frr(y, s, threshold: float) -> tuple[float, float]:
    """(FAR, FRR) em um threshold fixo; score == threshold é aceito."""
    pos, neg = _split(y, s)
    return float(np.mean(neg >= threshold)), float(np.mean(pos < threshold))


def _curves(y, s) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thresholds candidatos (os próprios scores, crescentes), FAR, FRR, TP e TN em cada um."""
    pos, neg = _split(y, s)
    cand = np.unique(np.asarray(s, dtype=np.float64))
    n_pos_below = np.searchsorted(pos, cand, side="left")  # positivos rejeitados
    n_neg_below = np.searchsorted(neg, cand, side="left")  # negativos rejeitados
    far = 1.0 - n_neg_below / len(neg)
    frr = n_pos_below / len(pos)
    return cand, far, frr, len(pos) - n_pos_below, n_neg_below


def roc_auc_eer(y, s) -> tuple[float, float, float]:
    """(ROC-AUC, EER, threshold no ponto de EER)."""
    from sklearn.metrics import roc_auc_score

    cand, far, frr, _, _ = _curves(y, s)
    i = int(np.argmin(np.abs(far - frr)))
    auc = float(roc_auc_score(np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)))
    return auc, float((far[i] + frr[i]) / 2.0), float(cand[i])


def select_threshold(y, s, criterion: str = "accuracy") -> float:
    """Escolhe o threshold. Use SOMENTE com os scores do conjunto de validação.

    - "accuracy": maximiza a acurácia; em empate, o menor threshold.
    - "eer": o threshold em que FAR e FRR mais se aproximam.
    """
    cand, far, frr, tp, tn = _curves(y, s)
    if criterion == "accuracy":
        return float(cand[int(np.argmax((tp + tn) / len(np.asarray(y))))])
    if criterion == "eer":
        return float(cand[int(np.argmin(np.abs(far - frr)))])
    raise ValueError(f"critério de threshold desconhecido: {criterion!r}")


def evaluate_at_threshold(y, s, threshold: float) -> dict:
    """Métricas completas em um threshold fixo (positivo = mesma pessoa)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    _split(y, s)  # valida as duas classes
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auc, eer, eer_threshold = roc_auc_eer(y, s)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": fp / (fp + tn),
        "frr": fn / (fn + tp),
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "n_pairs": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
    }
