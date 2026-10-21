"""Testes da spec 001 (docs/specs/001-face-detection.md). Dados sintéticos; sem fotos de pessoas."""
import numpy as np
import pytest
from PIL import Image

from facerec.face.detection import (
    Detection,
    NoFaceDetected,
    crop_face,
    eye_angle_degrees,
    rotate_points,
)


def blob_centroid(img: Image.Image, half: str) -> np.ndarray:
    """Centróide dos pixels claros na metade esquerda ou direita da imagem."""
    a = np.asarray(img.convert("L")).astype(float)
    w = a.shape[1]
    sub = a[:, : w // 2] if half == "left" else a[:, w // 2 :]
    ys, xs = np.nonzero(sub > 128)
    return np.array([xs.mean() + (0 if half == "left" else w // 2), ys.mean()])


def synthetic_face_with_tilted_eyes(left, right, size=200):
    img = Image.new("RGB", (size, size), (0, 0, 0))
    px = img.load()
    for cx, cy in (left, right):
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                px[cx + dx, cy + dy] = (255, 255, 255)
    return img


def test_eye_angle_sign_and_value():
    flat = np.array([[60, 100], [140, 100], [100, 130], [75, 160], [125, 160]], float)
    assert eye_angle_degrees(flat) == pytest.approx(0.0)
    tilted = flat.copy()
    tilted[1] = [140, 140]  # olho direito 40 px mais baixo, a 80 px de distância horizontal
    assert eye_angle_degrees(tilted) == pytest.approx(np.degrees(np.arctan(40 / 80)), abs=1e-6)
    assert eye_angle_degrees(tilted) > 0


def test_rotation_levels_the_eyes():
    left, right = (60, 90), (140, 130)  # olhos inclinados
    img = synthetic_face_with_tilted_eyes(left, right)
    lm = np.array([left, right, [100, 130], [80, 160], [120, 160]], float)
    det = Detection(box=(30.0, 60.0, 170.0, 190.0), landmarks=lm, confidence=0.99)
    out = crop_face(img, det, size=160, rotate=True)
    l_c, r_c = blob_centroid(out, "left"), blob_centroid(out, "right")
    assert abs(l_c[1] - r_c[1]) < 2.0  # olhos na mesma altura depois de alinhar
    unrotated = crop_face(img, det, size=160, rotate=False)
    assert abs(blob_centroid(unrotated, "left")[1] - blob_centroid(unrotated, "right")[1]) > 10


def test_rotate_points_matches_pil():
    img = synthetic_face_with_tilted_eyes((60, 90), (140, 130))
    center, angle = (100.0, 110.0), 25.0
    rotated = img.rotate(angle, resample=Image.NEAREST, center=center)
    expected = rotate_points(np.array([[60, 90], [140, 130]], float), center, angle)
    assert np.allclose(blob_centroid(rotated, "left"), expected[0], atol=1.5)
    assert np.allclose(blob_centroid(rotated, "right"), expected[1], atol=1.5)


def test_crop_size_and_margin():
    img = Image.new("RGB", (300, 300), (10, 20, 30))
    det = Detection(box=(100.0, 100.0, 200.0, 200.0), landmarks=np.zeros((5, 2)), confidence=1.0)
    assert crop_face(img, det, size=160).size == (160, 160)
    assert crop_face(img, det, size=112, margin_ratio=0.2).size == (112, 112)


def test_crop_outside_the_image_does_not_fail():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    det = Detection(box=(-20.0, -20.0, 60.0, 60.0), landmarks=np.zeros((5, 2)), confidence=1.0)
    assert crop_face(img, det).size == (160, 160)


def test_blank_image_has_no_face():  # spec 001, CA2
    pytest.importorskip("facenet_pytorch")
    from facerec.face.detection import FaceDetector

    detector = FaceDetector(min_confidence=0.9)
    blank = Image.new("RGB", (200, 200), (128, 128, 128))
    assert detector.detect(blank) == []
    with pytest.raises(NoFaceDetected):
        detector.detect_best(blank)
