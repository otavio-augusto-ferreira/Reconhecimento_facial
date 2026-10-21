"""Detector e gerador de embeddings falsos para testar a aplicação sem carregar modelos.

Convenção das imagens sintéticas (ver `make_png`):
- o pixel (0, 0) guarda a "legenda": R = número de faces, G = código de qualidade;
- cada face é um retângulo de cor uniforme e o canal R dessa cor é a "identidade" da face;
- o embedder falso devolve um vetor aleatório (semente = identidade): mesma identidade → similaridade 1,
  identidades diferentes → similaridade perto de 0.
"""
import io

import numpy as np
import pytest
from PIL import Image

from facerec.app.config import AppConfig
from facerec.app.service import FaceService
from facerec.app.storage import EMBEDDING_DIM, EmbeddingStore
from facerec.face.detection import Detection

GOOD, LOW_CONFIDENCE, SMALL_FACE = 0, 1, 2

_BOXES = {
    (1, GOOD): [(20, 20, 140, 140)],
    (1, LOW_CONFIDENCE): [(20, 20, 140, 140)],
    (1, SMALL_FACE): [(20, 20, 60, 60)],
}
_TWO_FACES = [(10, 10, 90, 90), (110, 110, 190, 190)]


def make_png(ids=(10,), quality=GOOD, size=(200, 200)) -> bytes:
    """PNG sintético com uma face por identidade em `ids` (lista vazia = imagem sem face)."""
    ids = list(ids)
    img = Image.new("RGB", size, (0, 0, 0))
    boxes = _TWO_FACES if len(ids) == 2 else _BOXES.get((len(ids), quality), [])
    for ident, box in zip(ids, boxes):
        img.paste((ident, 0, 0), box)
    img.putpixel((0, 0), (len(ids), quality, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def embedding_for(identity: int) -> np.ndarray:
    v = np.random.default_rng(identity).normal(size=EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeDetector:
    def detect(self, image):
        n, quality, _ = image.getpixel((0, 0))
        if n == 0:
            return []
        boxes = _TWO_FACES if n == 2 else _BOXES[(1, quality)]
        confidence = 0.95 if quality == LOW_CONFIDENCE else 0.99
        return [Detection(tuple(map(float, b)), np.zeros((5, 2)), confidence) for b in boxes]


class FakeEmbedder:
    def embed(self, faces):
        ids = [int(round(np.asarray(f)[..., 0].mean())) for f in faces]
        return np.stack([embedding_for(i) for i in ids])


@pytest.fixture
def config():
    return AppConfig(database_path=":memory:")


@pytest.fixture
def service(config):
    return FaceService(FakeDetector(), FakeEmbedder(), EmbeddingStore(":memory:"), config)
