"""API HTTP (spec 005): rotas, códigos de erro, interface estática e logs sem dados biométricos."""
import logging

import pytest
from fastapi.testclient import TestClient

from facerec.app.api import configure_logging, create_app
from conftest import LOW_CONFIDENCE, make_png


@pytest.fixture
def client(service):
    with TestClient(create_app(service=service)) as c:
        yield c


def png(ids=(10,), **kw):
    return ("foto.png", make_png(ids, **kw), "image/png")


def enroll(client, name="Ana", ids=(10,), n=1, **extra):
    return client.post("/api/persons", data={"name": name, **extra}, files=[("images", png(ids)) for _ in range(n)])


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["persons"] == 0 and body["enroll_max_images"] == 10


def test_lista_vazia_no_inicio(client):
    assert client.get("/api/persons").json() == {"persons": []}


def test_fluxo_cadastrar_reconhecer_excluir(client):
    r = enroll(client, n=2)
    assert r.status_code == 201
    person = r.json()["person"]
    assert person["name"] == "Ana" and person["n_images"] == 2

    rec = client.post("/api/recognize", files={"image": png()}).json()
    face = rec["faces"][0]
    assert face["known"] and face["name"] == "Ana"
    assert rec["image"] == {"width": 200, "height": 200}
    assert set(rec["timing_ms"]) == {"detect", "embed"} and len(face["box"]) == 4

    assert client.delete(f"/api/persons/{person['id']}").json() == {"deleted": person["id"]}
    assert client.get("/api/persons").json() == {"persons": []}
    after = client.post("/api/recognize", files={"image": png()}).json()["faces"][0]
    assert after["known"] is False and after["score"] is None


def test_desconhecido(client):
    enroll(client)
    face = client.post("/api/recognize", files={"image": png([99])}).json()["faces"][0]
    assert face["known"] is False and face["name"] is None and face["score"] is not None


@pytest.mark.parametrize(
    "files, status, code",
    [
        ([("images", png([]))], 422, "no_face"),
        ([("images", png([10, 20]))], 422, "multiple_faces"),
        ([("images", png([10], quality=LOW_CONFIDENCE))], 422, "low_quality"),
        ([("images", ("x.png", b"lixo", "image/png"))], 400, "invalid_image"),
        ([], 422, "no_images"),
    ],
)
def test_erros_de_cadastro_tem_codigo_e_mensagem(client, files, status, code):
    r = client.post("/api/persons", data={"name": "Ana"}, files=files or None)
    assert r.status_code == status
    err = r.json()["error"]
    assert err["code"] == code and err["message"]


def test_nome_vazio_e_repetido(client):
    assert enroll(client, name="  ").json()["error"]["code"] == "invalid_name"
    assert enroll(client).status_code == 201
    r = enroll(client, name="ana", ids=(20,))
    assert r.status_code == 409 and r.json()["error"]["code"] == "name_taken"


def test_duplicado_pede_confirmacao(client):
    enroll(client)
    r = enroll(client, name="Outra")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "duplicate_person" and err["existing_name"] == "Ana"
    assert enroll(client, name="Outra", confirm_duplicate="true").status_code == 201


def test_campos_ausentes(client):
    r = client.post("/api/persons", files=[("images", png())])  # sem nome
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    r = client.post("/api/recognize")  # sem imagem
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_reconhecer_sem_face(client):
    r = client.post("/api/recognize", files={"image": png([])})
    assert r.status_code == 422 and r.json()["error"]["code"] == "no_face"


def test_imagem_acima_do_limite(client):
    client.app.state.service.config = type(client.app.state.service.config)(max_image_bytes=50)
    r = client.post("/api/recognize", files={"image": png()})
    assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large"


def test_excluir_inexistente(client):
    r = client.delete("/api/persons/999")
    assert r.status_code == 404 and r.json()["error"]["code"] == "person_not_found"


def test_erro_inesperado_nao_vaza_detalhes(service):
    def boom(_):
        raise RuntimeError("detalhe interno secreto")

    service.recognize = boom
    with TestClient(create_app(service=service), raise_server_exceptions=False) as c:
        r = c.post("/api/recognize", files={"image": png()})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error" and "secreto" not in r.text


def test_interface_estatica_e_servida(client):
    r = client.get("/")
    assert r.status_code == 200 and "Reconhecedor Facial" in r.text
    assert client.get("/app.js").status_code == 200 and client.get("/styles.css").status_code == 200


def test_logs_nao_contem_nome_nem_dados_biometricos(service, tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    configure_logging(str(log_file))
    logger = logging.getLogger("facerec")
    try:
        with TestClient(create_app(service=service)) as c:
            pid = enroll(c, name="Fulana Secreta").json()["person"]["id"]
            c.post("/api/recognize", files={"image": png()})
            c.post("/api/recognize", files={"image": png([])})
            c.delete(f"/api/persons/{pid}")
        for h in logger.handlers:
            h.flush()
        text = log_file.read_text(encoding="utf-8")
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
    assert "enroll" in text and "recognize" in text and "delete" in text
    assert "Fulana" not in text and "Secreta" not in text
