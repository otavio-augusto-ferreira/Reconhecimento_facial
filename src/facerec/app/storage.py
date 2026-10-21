"""Armazenamento local de pessoas e embeddings em SQLite (ADR-002).

Privacidade por projeto:
- só guardamos embeddings (vetores), nunca as fotos enviadas;
- a exclusão de uma pessoa apaga suas linhas em cascata (ON DELETE CASCADE);
- `secure_delete` sobrescreve o conteúdo apagado e o VACUUM devolve o espaço ao sistema.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_persons_name ON persons(name COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);
"""


class NameAlreadyExists(ValueError):
    pass


@dataclass(frozen=True)
class Person:
    id: int
    name: str
    created_at: str
    n_embeddings: int


class EmbeddingStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: o FastAPI atende requisições em threads diferentes.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add_person(self, name: str, embeddings: np.ndarray) -> int:
        """Cadastra a pessoa e todos os seus embeddings numa única transação."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM or len(embeddings) == 0:
            raise ValueError(f"embeddings devem ter shape (n, {EMBEDDING_DIM}) com n >= 1")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._conn:
                cur = self._conn.execute("INSERT INTO persons (name, created_at) VALUES (?, ?)", (name, now))
                person_id = cur.lastrowid
                self._conn.executemany(
                    "INSERT INTO embeddings (person_id, vector) VALUES (?, ?)",
                    [(person_id, e.tobytes()) for e in embeddings],
                )
        except sqlite3.IntegrityError as exc:
            raise NameAlreadyExists(name) from exc
        return int(person_id)

    def list_persons(self) -> list[Person]:
        rows = self._conn.execute(
            "SELECT p.id, p.name, p.created_at, COUNT(e.id) FROM persons p "
            "LEFT JOIN embeddings e ON e.person_id = p.id GROUP BY p.id ORDER BY p.name COLLATE NOCASE"
        ).fetchall()
        return [Person(*r) for r in rows]

    def get_person(self, person_id: int) -> Person | None:
        return next((p for p in self.list_persons() if p.id == person_id), None)

    def all_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """(matriz (n, 512), ids das pessoas (n,)). Vazio se não há ninguém cadastrado."""
        rows = self._conn.execute("SELECT person_id, vector FROM embeddings ORDER BY id").fetchall()
        if not rows:
            return np.zeros((0, EMBEDDING_DIM), np.float32), np.zeros((0,), np.int64)
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return matrix, ids

    def delete_person(self, person_id: int) -> bool:
        """Remove a pessoa e, em cascata, todos os seus embeddings. False se não existir."""
        with self._conn:
            cur = self._conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        deleted = cur.rowcount > 0
        if deleted:
            self._conn.execute("VACUUM")  # compacta o arquivo; com secure_delete, o conteúdo é sobrescrito
        return deleted
