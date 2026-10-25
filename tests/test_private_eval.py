"""Avaliação agregada do dataset privado, com embeddings sintéticos (nenhuma foto)."""
import json

import numpy as np
import pytest

from facerec.private_eval import aggregate_verification, identification_simulation


def _people(n_persons=5, per_person=6, dim=32, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_persons, dim))
    emb, ids = [], []
    for p in range(n_persons):
        for _ in range(per_person):
            v = centers[p] + noise * rng.normal(size=dim)
            emb.append(v / np.linalg.norm(v))
            ids.append(f"p{p:02d}")
    return np.array(emb), ids


def test_pessoas_bem_separadas_sao_reconhecidas_e_desconhecidos_rejeitados():
    emb, ids = _people()
    r = identification_simulation(emb, ids, gallery_size=3, threshold=0.6)
    assert r["known_probes"]["correct_rate"] == 1.0 and r["known_probes"]["wrong_person"] == 0
    assert r["unknown_probes"]["false_accept_rate"] == 0.0
    assert r["n_persons"] == 5 and r["n_gallery_images"] == 15


def test_pessoas_identicas_geram_falso_aceite_e_troca_de_pessoa():
    emb, ids = _people(n_persons=4, per_person=6)
    emb = np.concatenate([emb[:6], emb[:6], emb[12:]])  # a pessoa 1 é cópia da pessoa 0
    r = identification_simulation(emb, ids, gallery_size=3, threshold=0.6)
    assert r["unknown_probes"]["false_accept_rate"] > 0.0


def test_threshold_alto_demais_rejeita_todos():
    emb, ids = _people()
    r = identification_simulation(emb, ids, gallery_size=3, threshold=1.01)
    assert r["known_probes"]["rejected_rate"] == 1.0


def test_exige_minimo_de_pessoas_e_ignora_quem_tem_poucas_fotos():
    emb, ids = _people(n_persons=2)
    with pytest.raises(ValueError, match="pelo menos 3 pessoas"):
        identification_simulation(emb, ids, gallery_size=3, threshold=0.5)
    emb, ids = _people(n_persons=4, per_person=4)
    r = identification_simulation(np.concatenate([emb, emb[:2]]), ids + ["extra", "extra"], gallery_size=3, threshold=0.5)
    assert r["n_persons"] == 4 and r["n_persons_skipped_few_photos"] == 1


def test_resultado_e_reprodutivel_e_nao_contem_nomes():
    emb, ids = _people()
    a = identification_simulation(emb, ids, 3, 0.5, seed=1)
    b = identification_simulation(emb, ids, 3, 0.5, seed=1)
    assert a == b
    text = json.dumps(a) + json.dumps(aggregate_verification(emb, ids, 0.5))
    assert not any(name in text for name in set(ids))


def test_verificacao_agregada_em_pares():
    emb, ids = _people()
    m = aggregate_verification(emb, ids, threshold=0.6)
    assert m["far"] == 0.0 and m["frr"] == 0.0 and m["roc_auc"] == 1.0 and m["n_pairs"] > 0
    q = m["score_quantiles"]
    assert q["mesma_pessoa"]["p50"] > q["pessoas_diferentes"]["p50"]
    assert q["mesma_pessoa"]["min"] <= q["mesma_pessoa"]["p50"] <= q["mesma_pessoa"]["max"]
