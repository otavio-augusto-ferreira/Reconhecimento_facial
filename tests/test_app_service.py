"""Regras de cadastro, reconhecimento e exclusão (specs 002, 003 e 004), com detector e modelo falsos."""
import dataclasses

import pytest

from facerec.app import errors as E
from facerec.app.service import FaceService
from facerec.app.storage import EmbeddingStore
from conftest import GOOD, LOW_CONFIDENCE, SMALL_FACE, FakeDetector, FakeEmbedder, make_png


# ------------------------------------------------------------------ cadastro (spec 002)
def test_cadastro_valido_cria_pessoa_e_embeddings(service):
    result = service.enroll("Ana", [make_png([10]), make_png([10])])
    assert result.person.name == "Ana" and result.person.n_embeddings == 2
    assert result.duplicate_confirmed is False
    assert [p.name for p in service.list_persons()] == ["Ana"]


def test_banco_recem_criado_nao_tem_ninguem(service):
    assert service.list_persons() == []  # spec 002, CA5


def test_nome_e_normalizado(service):
    assert service.enroll("  Ana   Maria ", [make_png([10])]).person.name == "Ana Maria"


@pytest.mark.parametrize("name", ["", "   ", None])
def test_nome_vazio_e_rejeitado(service, name):
    with pytest.raises(E.InvalidNameError):
        service.enroll(name, [make_png([10])])


def test_nome_longo_demais(service, config):
    with pytest.raises(E.InvalidNameError):
        service.enroll("x" * (config.max_name_length + 1), [make_png([10])])


def test_sem_imagens(service):
    with pytest.raises(E.NoImagesError):
        service.enroll("Ana", [])


def test_imagens_demais(service, config):
    with pytest.raises(E.TooManyImagesError):
        service.enroll("Ana", [make_png([10])] * (config.enroll_max_images + 1))


def test_imagem_invalida(service):
    with pytest.raises(E.InvalidImageError):
        service.enroll("Ana", [b"isto nao e uma imagem"])


def test_imagem_grande_demais(config):
    svc = FaceService(FakeDetector(), FakeEmbedder(), EmbeddingStore(":memory:"),
                      dataclasses.replace(config, max_image_bytes=100))
    with pytest.raises(E.ImageTooLargeError):
        svc.enroll("Ana", [make_png([10])])


def test_sem_face(service):
    with pytest.raises(E.NoFaceError):
        service.enroll("Ana", [make_png([])])
    assert service.list_persons() == []


def test_multiplas_faces(service):
    with pytest.raises(E.MultipleFacesError):
        service.enroll("Ana", [make_png([10, 20])])


@pytest.mark.parametrize("quality", [LOW_CONFIDENCE, SMALL_FACE])
def test_baixa_qualidade(service, quality):
    with pytest.raises(E.LowQualityError):
        service.enroll("Ana", [make_png([10], quality=quality)])


def test_erro_indica_qual_imagem_falhou(service):
    with pytest.raises(E.NoFaceError, match="Imagem 2"):
        service.enroll("Ana", [make_png([10]), make_png([])])
    assert service.list_persons() == []  # nada é salvo se uma imagem falha


def test_imagens_de_pessoas_diferentes_no_mesmo_cadastro(service):
    with pytest.raises(E.InconsistentImagesError):
        service.enroll("Ana", [make_png([10]), make_png([20])])


def test_nome_repetido_ignora_maiusculas(service):
    service.enroll("Ana", [make_png([10])])
    with pytest.raises(E.NameTakenError):
        service.enroll("ANA", [make_png([20])])


def test_cadastro_duplicado_exige_confirmacao(service):
    service.enroll("Ana", [make_png([10])])
    with pytest.raises(E.DuplicatePersonError) as info:
        service.enroll("Outra", [make_png([10])])
    assert info.value.extra["existing_name"] == "Ana"
    assert [p.name for p in service.list_persons()] == ["Ana"]
    result = service.enroll("Outra", [make_png([10])], confirm_duplicate=True)
    assert result.duplicate_confirmed is True and len(service.list_persons()) == 2


# ------------------------------------------------------------------ reconhecimento (specs 003 e 004)
def test_reconhece_pessoa_cadastrada(service):
    service.enroll("Ana", [make_png([10])])
    service.enroll("Bia", [make_png([20])])
    r = service.recognize(make_png([20]))
    [face] = r.faces
    assert face.known and face.name == "Bia" and face.score == pytest.approx(1.0, abs=1e-5)
    assert r.threshold == pytest.approx(0.5404)


def test_pessoa_nao_cadastrada_e_desconhecida(service):
    service.enroll("Ana", [make_png([10])])
    [face] = service.recognize(make_png([99])).faces
    assert not face.known and face.name is None and face.person_id is None
    assert face.score is not None and face.score < service.config.threshold


def test_banco_vazio_retorna_desconhecido_sem_erro(service):
    [face] = service.recognize(make_png([10])).faces
    assert not face.known and face.score is None  # spec 003, CA3


def test_duas_faces_na_mesma_imagem(service):
    service.enroll("Ana", [make_png([10])])
    r = service.recognize(make_png([10, 99]))
    assert [f.known for f in r.faces] == [True, False]
    assert r.faces[0].name == "Ana"


def test_reconhecimento_sem_face(service):
    with pytest.raises(E.NoFaceError):
        service.recognize(make_png([]))


def test_reconhecimento_registra_tempos(service):
    r = service.recognize(make_png([10]))
    assert r.detect_ms >= 0 and r.embed_ms >= 0  # spec 003, CA2


def test_limiar_vem_da_configuracao(config):
    strict = dataclasses.replace(config, threshold=1.01)  # ninguém passa
    svc = FaceService(FakeDetector(), FakeEmbedder(), EmbeddingStore(":memory:"), strict)
    svc.enroll("Ana", [make_png([10])], confirm_duplicate=True)
    [face] = svc.recognize(make_png([10])).faces
    assert not face.known


# ------------------------------------------------------------------ exclusão (spec 002, CA4)
def test_apos_excluir_a_pessoa_nao_e_mais_reconhecida(service):
    pid = service.enroll("Ana", [make_png([10])]).person.id
    assert service.recognize(make_png([10])).faces[0].known
    service.delete_person(pid)
    assert service.list_persons() == []
    [face] = service.recognize(make_png([10])).faces
    assert not face.known and face.score is None
    matrix, _ = service.store.all_embeddings()
    assert len(matrix) == 0


def test_excluir_pessoa_inexistente(service):
    with pytest.raises(E.PersonNotFoundError):
        service.delete_person(12345)


def test_pode_recadastrar_depois_de_excluir(service):
    pid = service.enroll("Ana", [make_png([10])]).person.id
    service.delete_person(pid)
    assert service.enroll("Ana", [make_png([10])]).person.name == "Ana"
