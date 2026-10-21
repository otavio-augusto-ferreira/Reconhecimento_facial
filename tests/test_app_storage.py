"""Armazenamento (ADR-002) e configuração da aplicação."""
import numpy as np
import pytest

from facerec.app.config import AppConfig, load_config
from facerec.app.storage import EMBEDDING_DIM, EmbeddingStore, NameAlreadyExists
from conftest import embedding_for


def test_banco_novo_esta_vazio(tmp_path):
    store = EmbeddingStore(tmp_path / "db" / "faces.db")
    assert store.list_persons() == []
    matrix, ids = store.all_embeddings()
    assert matrix.shape == (0, EMBEDDING_DIM) and ids.shape == (0,)


def test_cadastra_e_lista():
    store = EmbeddingStore(":memory:")
    pid = store.add_person("Ana", np.stack([embedding_for(1), embedding_for(2)]))
    [p] = store.list_persons()
    assert (p.id, p.name, p.n_embeddings) == (pid, "Ana", 2)
    matrix, ids = store.all_embeddings()
    assert matrix.shape == (2, EMBEDDING_DIM) and set(ids) == {pid}
    assert np.allclose(matrix[0], embedding_for(1))


def test_nome_unico_ignora_maiusculas():
    store = EmbeddingStore(":memory:")
    store.add_person("Ana", embedding_for(1)[None])
    with pytest.raises(NameAlreadyExists):
        store.add_person("ANA", embedding_for(2)[None])
    assert len(store.list_persons()) == 1  # a transação falha inteira, sem pessoa pela metade


@pytest.mark.parametrize("bad", [np.zeros((0, EMBEDDING_DIM)), np.zeros((1, 10)), np.zeros(EMBEDDING_DIM)])
def test_rejeita_formato_invalido(bad):
    store = EmbeddingStore(":memory:")
    with pytest.raises(ValueError):
        store.add_person("Ana", bad)
    assert store.list_persons() == []


def test_exclusao_remove_pessoa_e_embeddings():
    store = EmbeddingStore(":memory:")
    a = store.add_person("Ana", embedding_for(1)[None])
    b = store.add_person("Bia", embedding_for(2)[None])
    assert store.delete_person(a) is True
    assert [p.name for p in store.list_persons()] == ["Bia"]
    _, ids = store.all_embeddings()
    assert set(ids) == {b}
    assert store.delete_person(a) is False  # já não existe


def test_exclusao_apaga_os_bytes_do_arquivo(tmp_path):
    """spec 002, CA4: depois de excluir, nenhum vestígio do vetor fica no arquivo do banco."""
    path = tmp_path / "faces.db"
    store = EmbeddingStore(path)
    keep = store.add_person("Bia", embedding_for(2)[None])
    victim = store.add_person("Ana", embedding_for(1)[None])
    secret = embedding_for(1).tobytes()[:64]
    assert secret in path.read_bytes()  # confirma que o teste enxerga o vetor antes de excluir
    store.delete_person(victim)
    store.close()
    data = path.read_bytes()
    assert secret not in data
    assert embedding_for(2).tobytes()[:64] in data and keep  # quem ficou continua no banco


def test_config_do_repositorio_carrega():
    cfg = load_config("configs/app.yaml")
    assert cfg.threshold == pytest.approx(0.5404)
    assert cfg.effective_duplicate_threshold == cfg.threshold
    assert cfg.database_path == "database/facerec.db"


def test_config_usa_padroes_para_campos_ausentes(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("recognition:\n  threshold: 0.5\n  duplicate_threshold: 0.8\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.threshold == 0.5 and cfg.effective_duplicate_threshold == 0.8
    assert cfg.enroll_max_images == AppConfig().enroll_max_images
