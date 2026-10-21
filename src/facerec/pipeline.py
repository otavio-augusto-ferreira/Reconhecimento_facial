"""Pipeline imagem -> detecção -> recorte -> embedding, com cache em disco.

Spec: docs/specs/001-face-detection.md e docs/specs/003-face-recognition.md

O cache evita refazer a parte lenta (detecção e embedding) quando só mudam threshold ou métricas.
Ele é invalidado quando a `signature` (modelo, recorte, confiança mínima) muda.
Os arquivos de cache ficam em data/public/cache/, que o .gitignore já ignora.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from facerec.face.detection import NoFaceDetected, crop_face


@dataclass
class EmbeddingResult:
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # caminho -> vetor (512,)
    failures: dict[str, str] = field(default_factory=dict)  # caminho -> motivo
    seconds_detect: float = 0.0  # abrir imagem + detectar + recortar (soma)
    seconds_embed: float = 0.0  # rede neural (soma)
    n_timed: int = 0  # imagens que entraram nas somas acima


def _load_cache(cache_path: Path, signature: str) -> EmbeddingResult | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as z:
        if str(z["signature"]) != signature:
            return None
        paths, emb = z["paths"].tolist(), z["emb"]
        fail_paths, fail_reasons = z["fail_paths"].tolist(), z["fail_reasons"].tolist()
        t_detect, t_embed, n_timed = (float(v) for v in z["timing"])
    return EmbeddingResult(
        embeddings={p: emb[i] for i, p in enumerate(paths)},
        failures=dict(zip(fail_paths, fail_reasons)),
        seconds_detect=t_detect,
        seconds_embed=t_embed,
        n_timed=int(n_timed),
    )


def _save_cache(cache_path: Path, result: EmbeddingResult, signature: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    paths = list(result.embeddings)
    emb = np.stack([result.embeddings[p] for p in paths]) if paths else np.zeros((0, 512), np.float32)
    np.savez_compressed(
        cache_path,
        signature=np.array(signature),
        paths=np.array(paths, dtype=str),
        emb=emb,
        fail_paths=np.array(list(result.failures), dtype=str),
        fail_reasons=np.array(list(result.failures.values()), dtype=str),
        timing=np.array([result.seconds_detect, result.seconds_embed, result.n_timed], dtype=float),
    )


def compute_embeddings(
    paths: list[str],
    detector,
    embedder,
    *,
    size: int = 160,
    margin_ratio: float = 0.0,
    rotate: bool = False,
    batch_size: int = 32,
    cache_path: str | Path | None = None,
    signature: str = "",
    progress: bool = True,
) -> EmbeddingResult:
    """Embedding da face de maior confiança de cada imagem. Falhas são registradas, não levantadas."""
    paths = list(dict.fromkeys(paths))  # sem duplicatas, mantendo a ordem
    cache_path = Path(cache_path) if cache_path else None
    result = (_load_cache(cache_path, signature) if cache_path else None) or EmbeddingResult()
    todo = [p for p in paths if p not in result.embeddings and p not in result.failures]

    if todo:
        iterator = todo
        if progress:
            from tqdm import tqdm

            iterator = tqdm(todo, desc="detecção + embedding", unit="img")

        pending_paths: list[str] = []
        pending_faces: list[Image.Image] = []

        def flush() -> None:
            if not pending_faces:
                return
            t0 = time.perf_counter()
            vectors = embedder.embed(pending_faces)
            result.seconds_embed += time.perf_counter() - t0
            for p, v in zip(pending_paths, vectors):
                result.embeddings[p] = np.asarray(v, dtype=np.float32)
            pending_paths.clear()
            pending_faces.clear()

        for path in iterator:
            t0 = time.perf_counter()
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                detection = detector.detect_best(image)
                face = crop_face(image, detection, size=size, margin_ratio=margin_ratio, rotate=rotate)
            except NoFaceDetected:
                result.failures[path] = "no_face"
                face = None
            except (OSError, ValueError) as exc:  # arquivo corrompido ou ilegível
                result.failures[path] = f"error:{type(exc).__name__}"
                face = None
            result.seconds_detect += time.perf_counter() - t0
            result.n_timed += 1
            if face is not None:
                pending_paths.append(path)
                pending_faces.append(face)
                if len(pending_faces) >= batch_size:
                    flush()
        flush()
        if cache_path:
            _save_cache(cache_path, result, signature)

    # devolve só o que foi pedido (o cache pode conter mais imagens)
    wanted = set(paths)
    return EmbeddingResult(
        embeddings={p: v for p, v in result.embeddings.items() if p in wanted},
        failures={p: r for p, r in result.failures.items() if p in wanted},
        seconds_detect=result.seconds_detect,
        seconds_embed=result.seconds_embed,
        n_timed=result.n_timed,
    )
