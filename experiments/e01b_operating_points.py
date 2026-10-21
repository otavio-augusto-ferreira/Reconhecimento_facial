"""E01b — pontos de operação do FaceNet pré-treinado na LFW.

Mesmo modelo, detector e cache de embeddings do E01. A diferença: em vez de um único threshold
(acurácia máxima), compara quatro critérios, todos escolhidos SÓ na validação, e mostra o custo de
cada um no teste: acurácia máxima, EER, FAR <= 1% e FAR <= 0,1%.

Motivo (docs/decisions/ADR-003): na aplicação, aceitar uma pessoa desconhecida como se fosse cadastrada
é o erro mais grave, e com N pessoas cadastradas o risco cresce com N.

Uso (depois de rodar o E01, para reaproveitar o cache):
    python experiments/e01b_operating_points.py --config configs/e01b_operating_points.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from e01_facenet_baseline import SCORE_FUNCTIONS, limit_identities, versions

from facerec.data.splits import load_splits, make_verification_pairs
from facerec.metrics import evaluate_at_threshold, select_threshold
from facerec.pipeline import compute_embeddings


def expected_false_accepts(far: float, n_registered: int) -> float:
    """Chance de um desconhecido ser aceito por ALGUMA das N pessoas cadastradas, supondo pares independentes."""
    return 1.0 - (1.0 - far) ** n_registered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-identities", type=int, default=None, help="só para testes rápidos")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.cache_dir:
        cfg["cache_dir"] = args.cache_dir
    seed = cfg["seed"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from facerec.face.detection import FaceDetector
    from facerec.models.facenet import FaceNetEmbedder

    det_cfg, model_cfg = cfg["detector"], cfg["model"]
    detector = FaceDetector(det_cfg["min_confidence"], cfg["device"], det_cfg["min_face_size"])
    embedder = FaceNetEmbedder(model_cfg["pretrained"], cfg["device"])
    # Mesma assinatura do E01: se o detector/modelo mudarem, o cache é recalculado.
    signature = "|".join(
        str(v)
        for v in (
            model_cfg["pretrained"],
            det_cfg["min_confidence"],
            det_cfg["min_face_size"],
            det_cfg["margin_ratio"],
            det_cfg["rotate_to_eyes"],
        )
    )

    splits = load_splits(cfg["splits_manifest"], cfg["image_root"])
    scores, labels, counts = {}, {}, {}
    for k, name in enumerate(("val", "test")):
        items = splits[name]
        if args.limit_identities:
            items = limit_identities(items, args.limit_identities)
        result = compute_embeddings(
            [p for p, _ in items],
            detector,
            embedder,
            margin_ratio=det_cfg["margin_ratio"],
            rotate=det_cfg["rotate_to_eyes"],
            batch_size=model_cfg["batch_size"],
            cache_path=Path(cfg["cache_dir"]) / f"e01_{name}{'_limited' if args.limit_identities else ''}.npz",
            signature=signature,
        )
        ok_items = [(p, i) for p, i in items if p in result.embeddings]
        pairs = make_verification_pairs(ok_items, n_pairs=cfg["pairs"][name], seed=seed + k)
        index = {p: n for n, p in enumerate(result.embeddings)}
        matrix = np.stack(list(result.embeddings.values()))
        idx_a = np.array([index[a] for a, _, _ in pairs])
        idx_b = np.array([index[b] for _, b, _ in pairs])
        labels[name] = np.array([y for _, _, y in pairs])
        scores[name] = SCORE_FUNCTIONS[cfg["score"]](matrix, idx_a, idx_b)
        counts[name] = {
            "images_embedded": len(ok_items),
            "pairs": len(pairs),
            "positive": int(labels[name].sum()),
            "negative": int((1 - labels[name]).sum()),
        }

    rows = []
    for op in cfg["operating_points"]:
        thr = select_threshold(labels["val"], scores["val"], op["criterion"], op.get("far_target"))
        val = evaluate_at_threshold(labels["val"], scores["val"], thr)
        test = evaluate_at_threshold(labels["test"], scores["test"], thr)
        rows.append(
            {
                "name": op["name"],
                "criterion": op["criterion"],
                "far_target": op.get("far_target"),
                "threshold": thr,
                "val": {k: val[k] for k in ("far", "frr", "accuracy", "f1", "confusion")},
                "test": {k: test[k] for k in ("far", "frr", "accuracy", "f1", "confusion")},
                "expected_unknown_accepted_test": {
                    str(n): expected_false_accepts(test["far"], n) for n in cfg["registered_people"]
                },
            }
        )

    report = {
        "experiment": cfg["experiment"],
        "limited_to_identities": args.limit_identities,
        "config": cfg,
        "versions": versions(),
        "counts": counts,
        "operating_points": rows,
        "note": "Thresholds escolhidos só na validação. FAR do teste com poucos negativos tem incerteza grande; veja os números absolutos em confusion.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {cfg['experiment']} | pares: validação {counts['val']['pairs']}, teste {counts['test']['pairs']}"
          f" (negativos: {counts['val']['negative']} / {counts['test']['negative']})")
    head = f"{'ponto de operação':18}{'threshold':>10}{'val FAR':>9}{'val FRR':>9}{'teste FAR':>11}{'teste FRR':>11}{'falsos aceites':>16}"
    print(head)
    for r in rows:
        c = r["test"]["confusion"]
        print(
            f"{r['name']:18}{r['threshold']:10.4f}{r['val']['far']:9.4f}{r['val']['frr']:9.4f}"
            f"{r['test']['far']:11.4f}{r['test']['frr']:11.4f}{c['fp']:>8} de {c['fp'] + c['tn']:<5}"
        )
    print("\nchance de um desconhecido ser aceito por alguma das N pessoas cadastradas (FAR do teste):")
    print(f"{'ponto de operação':18}" + "".join(f"{'N=' + str(n):>9}" for n in cfg["registered_people"]))
    for r in rows:
        print(f"{r['name']:18}" + "".join(f"{r['expected_unknown_accepted_test'][str(n)]:9.3f}" for n in cfg["registered_people"]))
    print(f"\nresultados em {out_dir}/")


if __name__ == "__main__":
    main()
