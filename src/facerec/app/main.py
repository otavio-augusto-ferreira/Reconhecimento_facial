"""Ponto de entrada: `uvicorn facerec.app.main:app --host 127.0.0.1 --port 8000` (a partir da raiz do projeto)."""
from facerec.app.api import create_app

app = create_app()
