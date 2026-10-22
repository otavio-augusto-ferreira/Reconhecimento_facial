"""Cache de recortes de face e tabela de pontos de operação."""
import numpy as np
import pytest
from PIL import Image

from facerec.crops import compute_crops
from facerec.evaluation import expected_false_accepts, format_operating_points, operating_points
from facerec.face.detection import Detection, NoFaceDetected


class FakeDetector:
    """Imagem toda preta = sem face; senão a face é a metade esquerda da imagem."""

    def __init__(self):
        self.calls = 0

    def detect_best(self, image):
        self.calls += 1
        if np.asarray(image).max() == 0:
            raise NoFaceDetected("sem face")
        w, h = image.size
        return Detection((0.0, 0.0, w / 2, float(h)), np.zeros((5, 2)), 1.0)


def _images(tmp_path):
    paths = []
    for name, color in (("a", (200, 10, 10)), ("b", (10, 200, 10)), ("preta", (0, 0, 0))):
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (40, 20), color).save(p)
        paths.append(str(p))
    return paths


def test_recortes_falhas_e_ordem(tmp_path):
    a, b, preta = _images(tmp_path)
    res = compute_crops([b, preta, a], FakeDetector(), size=16, progress=False)
    assert res.paths == [b, a] and res.crops.shape == (2, 16, 16, 3) and res.crops.dtype == np.uint8
    assert res.failures == {preta: "no_face"}
    assert res.crops[0][..., 1].mean() > 150 and res.crops[1][..., 0].mean() > 150  # b é verde, a é vermelha


def test_cache_nao_repete_a_deteccao_e_aceita_imagens_novas(tmp_path):
    a, b, preta = _images(tmp_path)
    cache = tmp_path / "cache" / "crops.npz"
    det = FakeDetector()
    compute_crops([a, preta], det, size=16, cache_path=cache, signature="s1", progress=False)
    assert det.calls == 2
    again = compute_crops([a, preta], det, size=16, cache_path=cache, signature="s1", progress=False)
    assert det.calls == 2 and again.paths == [a] and again.failures == {preta: "no_face"}  # tudo veio do cache
    more = compute_crops([a, b], det, size=16, cache_path=cache, signature="s1", progress=False)
    assert det.calls == 3 and more.paths == [a, b]  # só a imagem nova foi processada


def test_cache_e_invalidado_quando_a_assinatura_muda(tmp_path):
    a, _, _ = _images(tmp_path)
    cache = tmp_path / "crops.npz"
    det = FakeDetector()
    compute_crops([a], det, size=16, cache_path=cache, signature="s1", progress=False)
    compute_crops([a], det, size=16, cache_path=cache, signature="s2", progress=False)
    assert det.calls == 2


def test_imagem_ilegivel_vira_falha_sem_derrubar_o_lote(tmp_path):
    a, _, _ = _images(tmp_path)
    ruim = tmp_path / "ruim.png"
    ruim.write_bytes(b"nao e uma imagem")
    res = compute_crops([str(ruim), a], FakeDetector(), size=16, progress=False)
    assert res.paths == [a] and res.failures[str(ruim)].startswith("error:")


def test_pontos_de_operacao_ficam_mais_rigidos_com_far_menor():
    rng = np.random.default_rng(0)
    y = np.array([1] * 3000 + [0] * 3000)
    s = np.concatenate([rng.normal(0.7, 0.15, 3000), rng.normal(0.2, 0.15, 3000)])
    labels, scores = {"val": y, "test": y}, {"val": s, "test": s}
    points = [
        {"name": "eer", "criterion": "eer"},
        {"name": "far_1pct", "criterion": "far", "far_target": 0.01},
        {"name": "far_0.1pct", "criterion": "far", "far_target": 0.001},
    ]
    rows = operating_points(labels, scores, points, [5, 10])
    thr = [r["threshold"] for r in rows]
    assert thr[0] < thr[1] < thr[2]
    assert rows[1]["val"]["far"] <= 0.01 and rows[2]["val"]["far"] <= 0.001
    assert rows[2]["val"]["frr"] > rows[1]["val"]["frr"] > rows[0]["val"]["frr"]
    assert set(rows[0]["expected_unknown_accepted_test"]) == {"5", "10"}
    text = format_operating_points(rows, [5, 10])
    assert "far_0.1pct" in text and "N=10" in text


def test_chance_de_aceitar_desconhecido_cresce_com_n():
    assert expected_false_accepts(0.0, 10) == 0.0
    assert expected_false_accepts(0.01, 10) == pytest.approx(1 - 0.99**10)
    assert expected_false_accepts(0.01, 20) > expected_false_accepts(0.01, 10)
