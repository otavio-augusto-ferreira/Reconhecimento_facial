"""Configuração da aplicação (spec 005). Lida de um YAML externo; nada é fixo no código."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "configs/app.yaml"


@dataclass(frozen=True)
class AppConfig:
    database_path: str = "database/facerec.db"
    log_path: str = "logs/app.log"
    device: str = "cpu"
    detector_min_confidence: float = 0.9
    detector_min_face_size: int = 20
    crop_size: int = 160
    crop_margin_ratio: float = 0.0
    crop_rotate_to_eyes: bool = False
    pretrained: str = "vggface2"
    checkpoint: str | None = None  # checkpoint fine-tunado (Parte 1); None = pesos pré-treinados
    threshold: float = 0.5404
    duplicate_threshold: float | None = None
    enroll_min_confidence: float = 0.97
    enroll_min_face_px: int = 80
    enroll_max_images: int = 10
    max_name_length: int = 80
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_side: int = 1280

    @property
    def effective_duplicate_threshold(self) -> float:
        return self.threshold if self.duplicate_threshold is None else self.duplicate_threshold


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    """Lê o YAML. Caminho: argumento, variável FACEREC_CONFIG ou configs/app.yaml."""
    path = Path(path or os.environ.get("FACEREC_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    det, crop = raw.get("detector", {}), raw.get("crop", {})
    rec, enr, img = raw.get("recognition", {}), raw.get("enrollment", {}), raw.get("image", {})
    d = AppConfig()
    return AppConfig(
        database_path=raw.get("database_path", d.database_path),
        log_path=raw.get("log_path", d.log_path),
        device=raw.get("device", d.device),
        detector_min_confidence=det.get("min_confidence", d.detector_min_confidence),
        detector_min_face_size=det.get("min_face_size", d.detector_min_face_size),
        crop_size=crop.get("size", d.crop_size),
        crop_margin_ratio=crop.get("margin_ratio", d.crop_margin_ratio),
        crop_rotate_to_eyes=crop.get("rotate_to_eyes", d.crop_rotate_to_eyes),
        pretrained=raw.get("model", {}).get("pretrained", d.pretrained),
        checkpoint=raw.get("model", {}).get("checkpoint", d.checkpoint),
        threshold=float(rec.get("threshold", d.threshold)),
        duplicate_threshold=rec.get("duplicate_threshold", d.duplicate_threshold),
        enroll_min_confidence=enr.get("min_confidence", d.enroll_min_confidence),
        enroll_min_face_px=enr.get("min_face_px", d.enroll_min_face_px),
        enroll_max_images=enr.get("max_images", d.enroll_max_images),
        max_name_length=enr.get("max_name_length", d.max_name_length),
        max_image_bytes=img.get("max_bytes", d.max_image_bytes),
        max_image_side=img.get("max_side", d.max_image_side),
    )
