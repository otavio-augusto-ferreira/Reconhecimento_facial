"""Avaliação AGREGADA no dataset privado. Roda SÓ localmente; nada daqui sai do computador.

Nada é ajustado nos dados privados: o threshold vem da validação da LFW (ADR-003). Os resultados guardam
apenas contagens e taxas, nunca nomes, caminhos, fotos nem embeddings (LGPD).
"""
from __future__ import annotations

import numpy as np

from facerec.data.splits import make_verification_pairs
from facerec.metrics import cosine_scores, evaluate_at_threshold

MIN_PERSONS = 3  # com menos pessoas, os números agregados quase identificam cada uma


def _unit(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float64)
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)


def _groups(identities: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, ident in enumerate(identities):
        groups.setdefault(ident, []).append(i)
    return groups


def aggregate_verification(emb, identities: list[str], threshold: float, seed: int = 42, n_pairs: int = 20000) -> dict:
    """Pares (mesma pessoa / pessoas diferentes) no threshold fixo da LFW. Devolve só métricas."""
    items = [(str(i), ident) for i, ident in enumerate(identities)]  # "caminho" = índice, nada identificável
    pairs = make_verification_pairs(items, n_pairs=n_pairs, seed=seed)
    a = np.array([int(x) for x, _, _ in pairs])
    b = np.array([int(y) for _, y, _ in pairs])
    y = np.array([lab for _, _, lab in pairs])
    s = cosine_scores(emb, a, b)
    m = evaluate_at_threshold(y, s, threshold)
    keys = ("accuracy", "precision", "recall", "f1", "far", "frr", "roc_auc", "eer", "confusion", "n_pairs", "n_positive", "n_negative")
    return {k: m[k] for k in keys}


def identification_simulation(emb, identities: list[str], gallery_size: int, threshold: float, seed: int = 42) -> dict:
    """Simula a aplicação: `gallery_size` fotos de cada pessoa são o "cadastro" e as demais são tentativas.

    - pessoas conhecidas: a tentativa é reconhecida corretamente, atribuída à pessoa errada ou rejeitada;
    - desconhecidos: cada pessoa, por vez, é retirada do cadastro e suas tentativas devem ser rejeitadas.
    """
    unit = _unit(emb)
    groups = {k: v for k, v in _groups(identities).items() if len(v) >= gallery_size + 1}
    skipped = len(set(identities)) - len(groups)
    if len(groups) < MIN_PERSONS:
        raise ValueError(
            f"são necessárias pelo menos {MIN_PERSONS} pessoas com {gallery_size + 1} ou mais fotos com face detectada "
            f"(há {len(groups)})"
        )
    rng = np.random.default_rng(seed)
    gallery: dict[str, np.ndarray] = {}
    probes: dict[str, np.ndarray] = {}
    for person in sorted(groups):
        idx = rng.permutation(groups[person])
        gallery[person], probes[person] = idx[:gallery_size], idx[gallery_size:]
    people = sorted(groups)

    def best_match(vec, allowed):
        best_person, best_score = None, -2.0
        for person in allowed:
            score = float((unit[gallery[person]] @ vec).max())
            if score > best_score:
                best_person, best_score = person, score
        return best_person, best_score

    correct = wrong = rejected = 0
    for person in people:
        for i in probes[person]:
            who, score = best_match(unit[i], people)
            if score < threshold:
                rejected += 1
            elif who == person:
                correct += 1
            else:
                wrong += 1
    n_known = correct + wrong + rejected

    accepted_unknown = n_unknown = 0
    for person in people:
        others = [p for p in people if p != person]
        for i in probes[person]:
            _, score = best_match(unit[i], others)
            n_unknown += 1
            accepted_unknown += int(score >= threshold)

    return {
        "n_persons": len(people),
        "n_persons_skipped_few_photos": skipped,
        "gallery_size": gallery_size,
        "n_gallery_images": int(sum(len(v) for v in gallery.values())),
        "threshold": threshold,
        "known_probes": {
            "n": n_known,
            "correct": correct,
            "wrong_person": wrong,
            "rejected": rejected,
            "correct_rate": correct / n_known,
            "wrong_person_rate": wrong / n_known,
            "rejected_rate": rejected / n_known,
        },
        "unknown_probes": {
            "n": n_unknown,
            "accepted": accepted_unknown,
            "false_accept_rate": accepted_unknown / n_unknown,
        },
    }
