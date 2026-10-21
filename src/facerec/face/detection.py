"""Detecção e alinhamento de faces.

Spec: docs/specs/001-face-detection.md

Usa o MTCNN do facenet-pytorch. A importação do PyTorch só acontece quando um `FaceDetector`
é criado; assim as funções de geometria (rotação e recorte) podem ser testadas sem carregar modelos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


class NoFaceDetected(ValueError):
    """Nenhuma face com confiança suficiente na imagem (spec 001, CA2)."""


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, em pixels
    landmarks: np.ndarray  # (5, 2): olho esquerdo, olho direito, nariz, canto esq. e dir. da boca
    confidence: float


class FaceDetector:
    """Detecta faces e devolve as detecções ordenadas por confiança (spec 001, CA3)."""

    def __init__(self, min_confidence: float = 0.9, device: str = "cpu", min_face_size: int = 20):
        from facenet_pytorch import MTCNN  # importação adiada (carrega o PyTorch)

        self.min_confidence = min_confidence  # spec 001, CA4: vem de configs/
        self._mtcnn = MTCNN(keep_all=True, device=device, min_face_size=min_face_size, post_process=False)

    def detect(self, image: Image.Image) -> list[Detection]:
        boxes, probs, points = self._mtcnn.detect(image.convert("RGB"), landmarks=True)
        if boxes is None:
            return []
        # o MTCNN devolve arrays de dtype object; convertemos para float
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        probs = np.asarray(probs, dtype=float).reshape(-1)
        points = np.asarray(points, dtype=float).reshape(-1, 5, 2)
        found = [
            Detection(tuple(map(float, b)), p, float(c))
            for b, p, c in zip(boxes, points, probs)
            if c >= self.min_confidence
        ]
        return sorted(found, key=lambda d: d.confidence, reverse=True)

    def detect_best(self, image: Image.Image) -> Detection:
        """A face de maior confiança. Levanta NoFaceDetected se não houver nenhuma."""
        found = self.detect(image)
        if not found:
            raise NoFaceDetected("nenhuma face detectada com a confiança mínima configurada")
        return found[0]


def eye_angle_degrees(landmarks: np.ndarray) -> float:
    """Inclinação da linha dos olhos, em graus, no sentido anti-horário (convenção do PIL).

    No sistema de coordenadas da imagem o eixo y cresce para baixo; se o olho direito está
    mais baixo que o esquerdo, o ângulo é positivo e girar a imagem por esse ângulo nivela os olhos.
    """
    (lx, ly), (rx, ry) = landmarks[0], landmarks[1]
    return math.degrees(math.atan2(ry - ly, rx - lx))


def rotate_points(points: np.ndarray, center: tuple[float, float], angle_degrees: float) -> np.ndarray:
    """Posição de pontos depois de `Image.rotate(angle_degrees, center=center)`."""
    phi = math.radians(angle_degrees)
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    x = rel[..., 0] * math.cos(phi) + rel[..., 1] * math.sin(phi)
    y = -rel[..., 0] * math.sin(phi) + rel[..., 1] * math.cos(phi)
    return np.stack([x, y], axis=-1) + np.asarray(center, dtype=float)


def crop_face(
    image: Image.Image,
    detection: Detection,
    size: int = 160,
    margin_ratio: float = 0.0,
    rotate: bool = False,
    resample: int = Image.BILINEAR,
) -> Image.Image:
    """Recorta a face e redimensiona para `size` x `size` (spec 001, R3).

    - margin_ratio: fração da largura/altura da caixa acrescentada de cada lado.
    - rotate: se True, gira a imagem para deixar a linha dos olhos horizontal antes de recortar.
    """
    x1, y1, x2, y2 = detection.box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    image = image.convert("RGB")

    if rotate:
        eyes_mid = tuple(detection.landmarks[:2].mean(axis=0))
        angle = eye_angle_degrees(detection.landmarks)
        image = image.rotate(angle, resample=Image.BICUBIC, center=eyes_mid)
        cx, cy = rotate_points(np.array([cx, cy]), eyes_mid, angle)

    half_w = w * (1.0 + 2.0 * margin_ratio) / 2.0
    half_h = h * (1.0 + 2.0 * margin_ratio) / 2.0
    box = (round(cx - half_w), round(cy - half_h), round(cx + half_w), round(cy + half_h))
    return image.crop(box).resize((size, size), resample)
