"""Erros da aplicação, com código estável e mensagem em português para a interface (spec 005, R4)."""
from __future__ import annotations


class AppError(Exception):
    code = "error"
    status = 400

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra


class InvalidImageError(AppError):
    code, status = "invalid_image", 400


class ImageTooLargeError(AppError):
    code, status = "image_too_large", 413


class NoFaceError(AppError):
    code, status = "no_face", 422


class MultipleFacesError(AppError):
    code, status = "multiple_faces", 422


class LowQualityError(AppError):
    code, status = "low_quality", 422


class InvalidNameError(AppError):
    code, status = "invalid_name", 422


class NoImagesError(AppError):
    code, status = "no_images", 422


class TooManyImagesError(AppError):
    code, status = "too_many_images", 422


class InconsistentImagesError(AppError):
    code, status = "inconsistent_images", 422


class NameTakenError(AppError):
    code, status = "name_taken", 409


class DuplicatePersonError(AppError):
    code, status = "duplicate_person", 409


class PersonNotFoundError(AppError):
    code, status = "person_not_found", 404
