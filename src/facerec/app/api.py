"""API HTTP (FastAPI) e entrega da interface estática (spec 005).

O backend não conhece a interface: ela só consome estas rotas. Tudo roda localmente.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from facerec.app.config import AppConfig, load_config
from facerec.app.errors import AppError, ImageTooLargeError
from facerec.app.service import FaceService
from facerec.app.storage import EmbeddingStore

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("facerec.app")


def configure_logging(path: str) -> None:
    """Log em arquivo local. Registra só eventos e contagens: nunca imagens, embeddings ou nomes."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("facerec")
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(handler)


def build_service(config: AppConfig) -> FaceService:
    """Monta o serviço com os modelos reais (importação adiada: carrega o PyTorch)."""
    from facerec.face.detection import FaceDetector
    from facerec.models.facenet import FaceNetEmbedder

    detector = FaceDetector(
        min_confidence=config.detector_min_confidence,
        device=config.device,
        min_face_size=config.detector_min_face_size,
    )
    embedder = FaceNetEmbedder(pretrained=config.pretrained, device=config.device)
    return FaceService(detector, embedder, EmbeddingStore(config.database_path), config)


def _person_json(p) -> dict:
    return {"id": p.id, "name": p.name, "created_at": p.created_at, "n_images": p.n_embeddings}


def create_app(service: FaceService | None = None, config: AppConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            cfg = config or load_config()
            configure_logging(cfg.log_path)
            app.state.service = build_service(cfg)
            log.info("aplicação iniciada")
        else:
            app.state.service = service
        yield

    app = FastAPI(title="Reconhecedor Facial", lifespan=lifespan)
    lock = threading.Lock()  # um pedido de cada vez no modelo e no SQLite

    def svc(request: Request) -> FaceService:
        return request.app.state.service

    def read_upload(request: Request, f: UploadFile) -> bytes:
        limit = svc(request).config.max_image_bytes
        data = f.file.read(limit + 1)
        if len(data) > limit:
            raise ImageTooLargeError(f"A imagem é maior que o limite de {limit // (1024 * 1024)} MB.")
        return data

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        body = {"code": exc.code, "message": exc.message, **exc.extra}
        return JSONResponse({"error": body}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        msg = "Requisição inválida: confira se todos os campos foram enviados."
        return JSONResponse({"error": {"code": "invalid_request", "message": msg}}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_handler(_: Request, exc: Exception):
        log.exception("erro inesperado")  # o traceback não contém imagens nem embeddings
        msg = "Erro interno. Tente novamente; se persistir, consulte o log da aplicação."
        return JSONResponse({"error": {"code": "internal_error", "message": msg}}, status_code=500)

    @app.get("/api/health")
    def health(request: Request):
        s = svc(request)
        return {
            "status": "ok",
            "persons": len(s.list_persons()),
            "threshold": s.config.threshold,
            "enroll_max_images": s.config.enroll_max_images,
        }

    @app.get("/api/persons")
    def list_persons(request: Request):
        with lock:
            return {"persons": [_person_json(p) for p in svc(request).list_persons()]}

    @app.post("/api/persons", status_code=201)
    def enroll(
        request: Request,
        name: str = Form(...),
        images: list[UploadFile] = File(default=[]),
        confirm_duplicate: bool = Form(False),
    ):
        data = [read_upload(request, f) for f in images]
        with lock:
            result = svc(request).enroll(name, data, confirm_duplicate)
        return {"person": _person_json(result.person), "duplicate_confirmed": result.duplicate_confirmed}

    @app.delete("/api/persons/{person_id}")
    def delete_person(request: Request, person_id: int):
        with lock:
            svc(request).delete_person(person_id)
        return {"deleted": person_id}

    @app.post("/api/recognize")
    def recognize(request: Request, image: UploadFile = File(...)):
        data = read_upload(request, image)
        with lock:
            r = svc(request).recognize(data)
        return {
            "image": {"width": r.image_width, "height": r.image_height},
            "threshold": r.threshold,
            "timing_ms": {"detect": round(r.detect_ms, 1), "embed": round(r.embed_ms, 1)},
            "faces": [
                {
                    "box": [round(v, 1) for v in f.box],
                    "known": f.known,
                    "person_id": f.person_id,
                    "name": f.name,
                    "score": None if f.score is None else round(f.score, 4),
                }
                for f in r.faces
            ],
        }

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
