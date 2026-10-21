"""Testes do pipeline de embeddings com detector e modelo falsos (sem PyTorch)."""
import numpy as np
from PIL import Image

from facerec.face.detection import Detection, NoFaceDetected
from facerec.pipeline import compute_embeddings


class FakeDetector:
    """Imagens totalmente pretas são 'sem face'; as outras têm uma face que ocupa tudo."""

    def __init__(self):
        self.calls = 0

    def detect_best(self, image):
        self.calls += 1
        if np.asarray(image).max() == 0:
            raise NoFaceDetected("sem face")
        w, h = image.size
        return Detection(box=(0.0, 0.0, float(w), float(h)), landmarks=np.zeros((5, 2)), confidence=1.0)


class FakeEmbedder:
    def __init__(self):
        self.batches = []

    def embed(self, faces):
        self.batches.append(len(faces))
        out = []
        for f in faces:
            v = np.zeros(512, np.float32)
            v[int(np.asarray(f).mean()) % 512] = 1.0
            out.append(v)
        return np.stack(out)


def make_images(tmp_path, n_good=5, n_black=2):
    paths = []
    for k in range(n_good):
        p = tmp_path / f"good{k}.png"
        Image.new("RGB", (64, 64), (10 + k, 10 + k, 10 + k)).save(p)
        paths.append(str(p))
    for k in range(n_black):
        p = tmp_path / f"black{k}.png"
        Image.new("RGB", (64, 64), (0, 0, 0)).save(p)
        paths.append(str(p))
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"isto nao e uma imagem")
    paths.append(str(corrupt))
    return paths


def test_failures_are_recorded_not_raised(tmp_path):
    paths = make_images(tmp_path)
    res = compute_embeddings(paths, FakeDetector(), FakeEmbedder(), batch_size=2, progress=False)
    assert len(res.embeddings) == 5
    assert sorted(res.failures.values()) == ["error:UnidentifiedImageError", "no_face", "no_face"]
    assert all(v.shape == (512,) for v in res.embeddings.values())


def test_batching(tmp_path):
    paths = make_images(tmp_path, n_good=5, n_black=0)
    emb = FakeEmbedder()
    compute_embeddings(paths[:5], FakeDetector(), emb, batch_size=2, progress=False)
    assert emb.batches == [2, 2, 1]


def test_cache_is_reused_and_invalidated_by_signature(tmp_path):
    paths = make_images(tmp_path)
    cache = tmp_path / "cache" / "x.npz"
    det1, emb1 = FakeDetector(), FakeEmbedder()
    first = compute_embeddings(paths, det1, emb1, cache_path=cache, signature="a", progress=False)
    assert det1.calls > 0

    det2, emb2 = FakeDetector(), FakeEmbedder()
    second = compute_embeddings(paths, det2, emb2, cache_path=cache, signature="a", progress=False)
    assert det2.calls == 0 and emb2.batches == []  # nada recalculado
    assert second.failures == first.failures
    assert all(np.array_equal(first.embeddings[p], second.embeddings[p]) for p in first.embeddings)

    det3 = FakeDetector()
    compute_embeddings(paths, det3, FakeEmbedder(), cache_path=cache, signature="b", progress=False)
    assert det3.calls > 0  # assinatura diferente invalida o cache


def test_cache_only_returns_requested_paths(tmp_path):
    paths = make_images(tmp_path)
    cache = tmp_path / "c.npz"
    compute_embeddings(paths, FakeDetector(), FakeEmbedder(), cache_path=cache, signature="s", progress=False)
    part = compute_embeddings(paths[:2], FakeDetector(), FakeEmbedder(), cache_path=cache, signature="s", progress=False)
    assert set(part.embeddings) | set(part.failures) == set(paths[:2])
