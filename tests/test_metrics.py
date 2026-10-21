"""Testes das métricas de verificação. Usam apenas dados sintéticos."""
import numpy as np
import pytest

from facerec.metrics import (
    cosine_scores,
    euclidean_scores,
    evaluate_at_threshold,
    far_frr,
    roc_auc_eer,
    select_threshold,
)


def test_far_frr_by_hand():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.6, 0.3, 0.7, 0.4, 0.2, 0.1])
    far, frr = far_frr(y, s, threshold=0.5)
    assert far == pytest.approx(1 / 4)  # só o negativo 0.7 foi aceito
    assert frr == pytest.approx(1 / 4)  # só o positivo 0.3 foi rejeitado


def test_threshold_is_inclusive():
    far, frr = far_frr(np.array([1, 0]), np.array([0.5, 0.5]), threshold=0.5)
    assert far == 1.0 and frr == 0.0  # score == threshold é aceito


def test_perfect_separation():
    y = np.array([1] * 50 + [0] * 50)
    s = np.concatenate([np.linspace(0.8, 1.0, 50), np.linspace(0.0, 0.5, 50)])
    thr = select_threshold(y, s, "accuracy")
    m = evaluate_at_threshold(y, s, thr)
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["roc_auc"] == 1.0
    assert m["far"] == 0.0 and m["frr"] == 0.0 and m["eer"] == 0.0
    assert 0.5 < thr <= 0.8


def test_random_scores_are_near_chance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    s = rng.random(4000)
    auc, eer, _ = roc_auc_eer(y, s)
    assert auc == pytest.approx(0.5, abs=0.05)
    assert eer == pytest.approx(0.5, abs=0.05)


def test_select_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.normal(loc=y * 1.0, scale=0.8)
    best = max(((np.mean((s >= t) == y.astype(bool)), -t) for t in np.unique(s)))
    assert select_threshold(y, s, "accuracy") == pytest.approx(-best[1])


def test_select_threshold_eer_balances_far_and_frr():
    rng = np.random.default_rng(2)
    y = np.array([1] * 500 + [0] * 500)
    s = np.concatenate([rng.normal(0.6, 0.15, 500), rng.normal(0.2, 0.15, 500)])
    thr = select_threshold(y, s, "eer")
    far, frr = far_frr(y, s, thr)
    assert abs(far - frr) < 0.03


def test_confusion_counts_add_up():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = evaluate_at_threshold(y, s, 0.5)
    c = m["confusion"]
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 2)
    assert m["n_pairs"] == 5 and m["n_positive"] == 2 and m["n_negative"] == 3


def test_cosine_and_euclidean_scores():
    emb = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    a, b = np.array([0, 0, 1]), np.array([2, 1, 2])
    assert cosine_scores(emb, a, b) == pytest.approx([1.0, 0.0, 0.0])  # independe da norma
    assert euclidean_scores(emb, a, b)[0] == pytest.approx(-2.0)


def test_single_class_is_rejected():
    with pytest.raises(ValueError):
        far_frr(np.array([1, 1]), np.array([0.1, 0.2]), 0.5)
    with pytest.raises(ValueError):
        select_threshold(np.array([0, 0]), np.array([0.1, 0.2]))


def test_select_threshold_far_respeita_o_alvo_com_o_menor_frr():
    rng = np.random.default_rng(3)
    y = np.array([1] * 2000 + [0] * 2000)
    s = np.concatenate([rng.normal(0.6, 0.15, 2000), rng.normal(0.2, 0.15, 2000)])
    thr = select_threshold(y, s, "far", far_target=0.01)
    far, frr = far_frr(y, s, thr)
    assert far <= 0.01
    # o threshold imediatamente menor já passaria do alvo: este é o menor possível
    smaller = np.max(s[s < thr])
    assert far_frr(y, s, smaller)[0] > 0.01
    # alvo mais rígido -> threshold maior e FRR maior
    thr2 = select_threshold(y, s, "far", far_target=0.001)
    assert thr2 > thr and far_frr(y, s, thr2)[1] >= frr


def test_select_threshold_far_exige_alvo_valido():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    with pytest.raises(ValueError):
        select_threshold(y, s, "far")
    with pytest.raises(ValueError):
        select_threshold(y, s, "far", far_target=1.5)
