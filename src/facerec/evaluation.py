"""Tabela de pontos de operação, compartilhada pelo E01b e pelos experimentos de fine-tuning.

Cada ponto escolhe o threshold SOMENTE na validação e o aplica, sem ajuste, ao teste
(docs/decisions/ADR-003). Assim todos os experimentos são comparáveis linha a linha.
"""
from __future__ import annotations

from facerec.metrics import evaluate_at_threshold, select_threshold


def expected_false_accepts(far: float, n_registered: int) -> float:
    """Chance de um desconhecido ser aceito por ALGUMA das N pessoas cadastradas, supondo pares independentes."""
    return 1.0 - (1.0 - far) ** n_registered


def operating_points(labels: dict, scores: dict, points: list[dict], registered_people: list[int]) -> list[dict]:
    """`labels` e `scores` têm as chaves "val" e "test". `points`: [{name, criterion, far_target?}, ...]."""
    rows = []
    for op in points:
        thr = select_threshold(labels["val"], scores["val"], op["criterion"], op.get("far_target"))
        val = evaluate_at_threshold(labels["val"], scores["val"], thr)
        test = evaluate_at_threshold(labels["test"], scores["test"], thr)
        keys = ("far", "frr", "accuracy", "f1", "confusion")
        rows.append(
            {
                "name": op["name"],
                "criterion": op["criterion"],
                "far_target": op.get("far_target"),
                "threshold": thr,
                "val": {k: val[k] for k in keys},
                "test": {k: test[k] for k in keys},
                "expected_unknown_accepted_test": {
                    str(n): expected_false_accepts(test["far"], n) for n in registered_people
                },
            }
        )
    return rows


def format_operating_points(rows: list[dict], registered_people: list[int]) -> str:
    head = (
        f"{'ponto de operação':18}{'threshold':>10}{'val FAR':>9}{'val FRR':>9}"
        f"{'teste FAR':>11}{'teste FRR':>11}{'falsos aceites':>16}"
    )
    lines = [head]
    for r in rows:
        c = r["test"]["confusion"]
        lines.append(
            f"{r['name']:18}{r['threshold']:10.4f}{r['val']['far']:9.4f}{r['val']['frr']:9.4f}"
            f"{r['test']['far']:11.4f}{r['test']['frr']:11.4f}{c['fp']:>8} de {c['fp'] + c['tn']:<5}"
        )
    lines.append("\nchance de um desconhecido ser aceito por alguma das N pessoas cadastradas (FAR do teste):")
    lines.append(f"{'ponto de operação':18}" + "".join(f"{'N=' + str(n):>9}" for n in registered_people))
    for r in rows:
        lines.append(
            f"{r['name']:18}"
            + "".join(f"{r['expected_unknown_accepted_test'][str(n)]:9.3f}" for n in registered_people)
        )
    return "\n".join(lines)
