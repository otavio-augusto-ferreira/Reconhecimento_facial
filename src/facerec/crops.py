"""Recortes de face (uint8) com cache em disco, usados no fine-tuning.

O fine-tuning repete as mesmas imagens por várias épocas; detectar e recortar uma única vez
(e guardar o resultado) evita refazer a parte lenta. Os arquivos ficam em data/public/cache/,
que o .gitignore já ignora, e só devem conter dados PÚBLICOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from facerec.face.detection import NoFaceDetected, crop_face


@dataclass
class CropResult:
    paths: list[str] = field(default_factory=list)  # imagens com face, na ordem pedida
    crops: np.ndarray = field(default_factory=lambda: np.zeros((0, 160, 160, 3), np.uint8))  # (n, size, size, 3)
    failures: dict[str, str] = field(default_factory=dict)  # caminho -> motivo
    seconds: float = 0.0  # tempo gasto detectando/recortando nesta chamada


def _load(cache_path: Path, signature: str):
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as z:
        if str(z["signature"]) != signature:
            return None
        return (
            z["paths"].tolist(),
            z["crops"],
            dict(zip(z["fail_paths"].tolist(), z["fail_reasons"].tolist())),
        )


def compute_crops(
    paths: list[str],
    detector,
    *,
    size: int = 160,
    margin_ratio: float = 0.0,
    rotate: bool = False,
    cache_path: str | Path | None = None,
    signature: str = "",
    progress: bool = True,
) -> CropResult:
    """Recorte da face de maior confiança de cada imagem. Falhas são registradas, não levantadas."""
    paths = list(dict.fromkeys(paths))
    cache_path = Path(cache_path) if cache_path else None
    cached = _load(cache_path, signature) if cache_path else None
    all_paths, all_crops, failures = cached or ([], np.zeros((0, size, size, 3), np.uint8), {})
    known = set(all_paths)
    todo = [p for p in paths if p not in known and p not in failures]

    seconds = 0.0
    if todo:
        iterator = todo
        if progress:
            from tqdm import tqdm

            iterator = tqdm(todo, desc="detecção + recorte", unit="img")
        new_paths: list[str] = []
        new_crops: list[np.ndarray] = []
        t0 = time.perf_counter()
        for path in iterator:
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                detection = detector.detect_best(image)
                face = crop_face(image, detection, size=size, margin_ratio=margin_ratio, rotate=rotate)
                new_crops.append(np.asarray(face.convert("RGB").resize((size, size)), dtype=np.uint8))
                new_paths.append(path)
            except NoFaceDetected:
                failures[path] = "no_face"
            except (OSError, ValueError) as exc:
                failures[path] = f"error:{type(exc).__name__}"
        seconds = time.perf_counter() - t0
        if new_paths:
            all_paths = list(all_paths) + new_paths
            all_crops = np.concatenate([all_crops, np.stack(new_crops)]) if len(all_crops) else np.stack(new_crops)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(  # sem compressão: o fine-tuning lê o arquivo inteiro várias vezes
                cache_path,
                signature=np.array(signature),
                paths=np.array(all_paths, dtype=str),
                crops=all_crops,
                fail_paths=np.array(list(failures), dtype=str),
                fail_reasons=np.array(list(failures.values()), dtype=str),
            )

    index = {p: i for i, p in enumerate(all_paths)}
    keep = [p for p in paths if p in index]
    crops = all_crops[[index[p] for p in keep]] if keep else np.zeros((0, size, size, 3), np.uint8)
    return CropResult(keep, crops, {p: r for p, r in failures.items() if p in set(paths)}, seconds)
