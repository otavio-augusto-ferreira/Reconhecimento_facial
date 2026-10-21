"""Regras de negócio da aplicação: cadastro, reconhecimento e exclusão.

Specs: 002 (cadastro/exclusão), 003 (reconhecimento), 004 (pessoa desconhecida).
O detector e o gerador de embeddings são injetados, o que permite testar sem carregar modelos.
Nenhuma foto é gravada em disco: a imagem enviada vive só em memória durante a requisição.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from facerec.app import errors as E
from facerec.app.config import AppConfig
from facerec.app.storage import EmbeddingStore, NameAlreadyExists, Person
from facerec.face.detection import Detection, crop_face

log = logging.getLogger("facerec.app")


@dataclass(frozen=True)
class FaceResult:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, em pixels da imagem analisada
    known: bool
    person_id: int | None
    name: str | None
    score: float | None  # similaridade cosseno com a pessoa mais parecida (None se não há ninguém cadastrado)


@dataclass(frozen=True)
class RecognitionResult:
    faces: list[FaceResult]
    image_width: int
    image_height: int
    threshold: float
    detect_ms: float
    embed_ms: float


@dataclass(frozen=True)
class EnrollResult:
    person: Person
    duplicate_confirmed: bool


class FaceService:
    def __init__(self, detector, embedder, store: EmbeddingStore, config: AppConfig):
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self.config = config

    # ------------------------------------------------------------------ imagens
    def decode_image(self, data: bytes) -> Image.Image:
        cfg = self.config
        if len(data) > cfg.max_image_bytes:
            raise E.ImageTooLargeError(f"A imagem é maior que o limite de {cfg.max_image_bytes // (1024 * 1024)} MB.")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as exc:  # arquivo corrompido, formato desconhecido, bomba de descompressão
            raise E.InvalidImageError("Não foi possível ler a imagem. Envie um arquivo JPEG ou PNG válido.") from exc
        if max(img.size) > cfg.max_image_side:
            scale = cfg.max_image_side / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
        return img

    def _crops(self, image: Image.Image, detections: list[Detection]) -> list[Image.Image]:
        c = self.config
        return [
            crop_face(image, d, size=c.crop_size, margin_ratio=c.crop_margin_ratio, rotate=c.crop_rotate_to_eyes)
            for d in detections
        ]

    # ------------------------------------------------------------------ cadastro
    def _clean_name(self, name: str) -> str:
        name = " ".join((name or "").split())
        if not name:
            raise E.InvalidNameError("Informe o nome da pessoa.")
        if len(name) > self.config.max_name_length:
            raise E.InvalidNameError(f"O nome deve ter no máximo {self.config.max_name_length} caracteres.")
        return name

    def enroll(self, name: str, images: list[bytes], confirm_duplicate: bool = False) -> EnrollResult:
        cfg = self.config
        name = self._clean_name(name)
        if not images:
            raise E.NoImagesError("Envie ao menos uma imagem com o rosto da pessoa.")
        if len(images) > cfg.enroll_max_images:
            raise E.TooManyImagesError(f"Envie no máximo {cfg.enroll_max_images} imagens por cadastro.")
        if any(p.name.casefold() == name.casefold() for p in self.store.list_persons()):
            raise E.NameTakenError("Já existe uma pessoa cadastrada com esse nome.")

        crops: list[Image.Image] = []
        for i, data in enumerate(images, start=1):
            prefix = f"Imagem {i}: " if len(images) > 1 else ""
            try:
                image = self.decode_image(data)
                detections = self.detector.detect(image)
                if not detections:
                    raise E.NoFaceError("Nenhuma face detectada. Use uma foto nítida, de frente e bem iluminada.")
                if len(detections) > 1:
                    raise E.MultipleFacesError(
                        f"{len(detections)} faces detectadas. No cadastro, a imagem deve ter apenas uma pessoa."
                    )
                det = detections[0]
                side = min(det.box[2] - det.box[0], det.box[3] - det.box[1])
                if det.confidence < cfg.enroll_min_confidence or side < cfg.enroll_min_face_px:
                    raise E.LowQualityError(
                        "Qualidade insuficiente. Aproxime o rosto da câmera e melhore a iluminação "
                        f"(face de pelo menos {cfg.enroll_min_face_px} px)."
                    )
            except E.AppError as exc:
                exc.message = prefix + exc.message
                exc.args = (exc.message,)
                raise
            crops.append(self._crops(image, [det])[0])

        embeddings = np.asarray(self.embedder.embed(crops), dtype=np.float32)

        if len(embeddings) > 1:
            # As imagens do mesmo cadastro devem ser da mesma pessoa: cada uma é comparada com a média
            # das OUTRAS (comparar com a média que inclui a própria imagem inflaria a similaridade).
            total = embeddings.sum(axis=0)
            for e in embeddings:
                others = total - e
                others /= np.linalg.norm(others) + 1e-12
                if float(e @ others) < cfg.threshold:
                    raise E.InconsistentImagesError(
                        "As imagens parecem ser de pessoas diferentes. Envie apenas fotos da mesma pessoa."
                    )

        duplicate = False
        stored, ids = self.store.all_embeddings()
        if len(stored):
            sims = embeddings @ stored.T  # (novas, armazenadas)
            j = int(np.unravel_index(np.argmax(sims), sims.shape)[1])
            score = float(sims.max())
            if score >= cfg.effective_duplicate_threshold:
                existing = self.store.get_person(int(ids[j]))
                if not confirm_duplicate:
                    raise E.DuplicatePersonError(
                        f"Este rosto é muito parecido com o de '{existing.name}', já cadastrado. "
                        "Confirme se deseja cadastrar mesmo assim.",
                        existing_name=existing.name,
                        score=round(score, 4),
                    )
                duplicate = True

        try:
            person_id = self.store.add_person(name, embeddings)
        except NameAlreadyExists as exc:
            raise E.NameTakenError("Já existe uma pessoa cadastrada com esse nome.") from exc
        log.info("enroll person_id=%d images=%d duplicate_confirmed=%s", person_id, len(images), duplicate)
        return EnrollResult(self.store.get_person(person_id), duplicate)

    # ------------------------------------------------------------------ reconhecimento
    def recognize(self, data: bytes) -> RecognitionResult:
        cfg = self.config
        image = self.decode_image(data)
        t0 = time.perf_counter()
        detections = self.detector.detect(image)
        t1 = time.perf_counter()
        if not detections:
            log.info("recognize faces=0")
            raise E.NoFaceError("Nenhuma face detectada. Posicione o rosto na frente da câmera, com boa iluminação.")
        embeddings = np.asarray(self.embedder.embed(self._crops(image, detections)), dtype=np.float32)
        t2 = time.perf_counter()

        stored, ids = self.store.all_embeddings()
        names = {p.id: p.name for p in self.store.list_persons()} if len(stored) else {}
        faces: list[FaceResult] = []
        for det, emb in zip(detections, embeddings):
            if len(stored) == 0:  # banco vazio: ninguém é conhecido, sem erro (spec 003, CA3)
                faces.append(FaceResult(det.box, False, None, None, None))
                continue
            sims = stored @ emb
            best_id, best = None, -2.0
            for pid in np.unique(ids):
                s = float(sims[ids == pid].max())
                if s > best:
                    best_id, best = int(pid), s
            known = best >= cfg.threshold
            faces.append(
                FaceResult(det.box, known, best_id if known else None, names[best_id] if known else None, best)
            )

        result = RecognitionResult(
            faces, image.width, image.height, cfg.threshold, (t1 - t0) * 1000, (t2 - t1) * 1000
        )
        log.info(
            "recognize faces=%d known=%d unknown=%d detect_ms=%.0f embed_ms=%.0f",
            len(faces), sum(f.known for f in faces), sum(not f.known for f in faces),
            result.detect_ms, result.embed_ms,
        )
        return result

    # ------------------------------------------------------------------ pessoas
    def list_persons(self) -> list[Person]:
        return self.store.list_persons()

    def delete_person(self, person_id: int) -> None:
        if not self.store.delete_person(person_id):
            raise E.PersonNotFoundError("Pessoa não encontrada.")
        log.info("delete person_id=%d", person_id)
