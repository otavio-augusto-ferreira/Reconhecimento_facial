"""Divisão por identidade, pares de verificação e checagem de leakage.

Spec: docs/specs/006-dataset-splits.md

Um "item" é uma tupla (caminho_da_imagem, identidade).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

Item = tuple[str, str]
Splits = dict[str, list[Item]]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")


class LeakageError(ValueError):
    """Levantada quando treino/validação/teste compartilham informação."""


def scan_identity_folders(root: str | Path) -> list[Item]:
    """Lista imagens sob `root`; a identidade é o nome da pasta que contém a imagem."""
    root = Path(root)
    items: list[Item] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.parent != root:
            items.append((str(path), path.parent.name))
    return items


def _group_by_identity(items: list[Item]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path, identity in items:
        groups[identity].append(path)
    return {identity: sorted(paths) for identity, paths in groups.items()}


def filter_min_images(
    items: list[Item], min_images: int = 2, max_images: int | None = None, seed: int = 42
) -> list[Item]:
    """Mantém identidades com >= min_images imagens.

    Se max_images for dado, limita cada identidade (amostragem determinística),
    para que poucas pessoas muito fotografadas não dominem o conjunto.
    """
    rng = random.Random(seed)
    kept: list[Item] = []
    groups = _group_by_identity(items)
    for identity in sorted(groups):
        paths = groups[identity]
        if len(paths) < min_images:
            continue
        if max_images is not None and len(paths) > max_images:
            paths = sorted(rng.sample(paths, max_images))
        kept.extend((p, identity) for p in paths)
    return kept


def split_by_identity(
    items: list[Item], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> Splits:
    """Divide as *identidades* (não as imagens) em treino, validação e teste."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("as proporções devem somar 1")
    groups = _group_by_identity(items)
    identities = sorted(groups)
    if len(identities) < 3:
        raise ValueError("são necessárias pelo menos 3 identidades")

    random.Random(seed).shuffle(identities)
    n = len(identities)
    # val e teste têm pelo menos 1 identidade; treino fica com o restante
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("proporções deixam o conjunto de treino vazio")
    chosen = {
        "train": identities[:n_train],
        "val": identities[n_train : n_train + n_val],
        "test": identities[n_train + n_val :],
    }
    return {
        name: [(path, identity) for identity in ids for path in groups[identity]]
        for name, ids in chosen.items()
    }


def make_verification_pairs(
    items: list[Item], n_pairs: int, seed: int = 42
) -> list[tuple[str, str, int]]:
    """Gera pares (img_a, img_b, rótulo): 1 = mesma pessoa, 0 = pessoas diferentes.

    Os pares são balanceados. Se não houver pares positivos distintos suficientes,
    gera menos pares, mantendo a proporção 50/50.
    """
    rng = random.Random(seed)
    groups = _group_by_identity(items)
    identities = sorted(groups)
    multi = [i for i in identities if len(groups[i]) >= 2]
    if not multi or len(identities) < 2:
        raise ValueError("são necessárias >= 2 identidades, ao menos uma com >= 2 imagens")

    target = n_pairs // 2
    positives: set[tuple[str, str]] = set()
    attempts = 0
    while len(positives) < target and attempts < target * 50:
        attempts += 1
        a, b = sorted(rng.sample(groups[rng.choice(multi)], 2))
        positives.add((a, b))

    negatives: set[tuple[str, str]] = set()
    while len(negatives) < len(positives):
        i, j = rng.sample(identities, 2)
        a, b = rng.choice(groups[i]), rng.choice(groups[j])
        negatives.add(tuple(sorted((a, b))))  # type: ignore[arg-type]

    return [(a, b, 1) for a, b in sorted(positives)] + [(a, b, 0) for a, b in sorted(negatives)]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_cross_split_duplicates(splits: Splits) -> list[tuple[str, str]]:
    """Pares de arquivos com conteúdo idêntico em conjuntos diferentes."""
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[tuple[str, str]] = []
    for name in sorted(splits):
        for path, _ in splits[name]:
            digest = _sha256(path)
            if digest in seen and seen[digest][0] != name:
                duplicates.append((seen[digest][1], path))
            seen.setdefault(digest, (name, path))
    return duplicates


def check_no_leakage(splits: Splits, check_hashes: bool = True) -> None:
    """Levanta LeakageError se houver identidade, caminho ou arquivo em mais de um conjunto."""
    names = sorted(splits)
    for idx, a in enumerate(names):
        for b in names[idx + 1 :]:
            shared_ids = {i for _, i in splits[a]} & {i for _, i in splits[b]}
            if shared_ids:
                raise LeakageError(f"identidades em '{a}' e '{b}': {sorted(shared_ids)[:5]}")
            shared_paths = {p for p, _ in splits[a]} & {p for p, _ in splits[b]}
            if shared_paths:
                raise LeakageError(f"caminhos em '{a}' e '{b}': {sorted(shared_paths)[:5]}")
    if check_hashes:
        dups = find_cross_split_duplicates(splits)
        if dups:
            raise LeakageError(f"arquivos idênticos entre conjuntos: {dups[:5]}")


def save_splits(splits: Splits, out: str | Path, root: str | Path, seed: int) -> None:
    """Salva o manifesto com caminhos relativos a `root` (reprodutibilidade)."""
    root = Path(root)
    data = {
        "seed": seed,
        "splits": {
            name: [[str(Path(p).relative_to(root)), i] for p, i in items]
            for name, items in splits.items()
        },
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def load_splits(path: str | Path, root: str | Path) -> Splits:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    root = Path(root)
    return {
        name: [(str(root / rel), identity) for rel, identity in items]
        for name, items in data["splits"].items()
    }
