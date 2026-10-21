"""E01 — FaceNet pré-treinado (VGGFace2), sem fine-tuning, avaliado em pares de verificação do LFW.

Protocolo (docs/experiments/E01-facenet-baseline.md):
  1. detecta e recorta a face de cada imagem de validação e de teste (MTCNN);
  2. gera embeddings (FaceNet pré-treinado);
  3. monta pares balanceados (mesma pessoa / pessoas diferentes) em cada conjunto;
  4. escolhe o THRESHOLD só na validação;
  5. avalia no teste, uma única vez, com esse threshold.

Uso (na raiz do repositório):
    python experiments/e01_facenet_baseline.py --config configs/e01_facenet_baseline.yaml
Teste rápido, com poucas pessoas:
    python experiments/e01_facenet_baseline.py --config configs/e01_facenet_baseline.yaml --limit-identities 20
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import yaml

from facerec.data.splits import load_splits, make_verification_pairs
from facerec.metrics import (
    cosine_scores,
    euclidean_scores,
    evaluate_at_threshold,
    roc_auc_eer,
    select_threshold,
)
from facerec.pipeline import compute_embeddings

SCORE_FUNCTIONS = {"cosine": cosine_scores, "euclidean": euclidean_scores}


def limit_identities(items, n):
    """Mantém só as n primeiras identidades (para testes rápidos)."""
    keep = sorted({i for _, i in items})[:n]
    return [(p, i) for p, i in items if i in keep]


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "torchvision", "facenet-pytorch", "numpy", "scikit-learn", "Pillow"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "não instalado"
    return out


def plot_results(out_dir: Path, scores: dict, labels: dict, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes[:2], ("val", "test")):
        s, y = scores[name], labels[name].astype(bool)
        bins = np.linspace(s.min(), s.max(), 50)
        ax.hist(s[y], bins=bins, alpha=0.6, label="mesma pessoa")
        ax.hist(s[~y], bins=bins, alpha=0.6, label="pessoas diferentes")
        ax.axvline(threshold, color="k", linestyle="--", label=f"threshold = {threshold:.3f}")
        ax.set_title(f"Distribuição de similaridade ({'validação' if name == 'val' else 'teste'})")
        ax.set_xlabel("similaridade")
        ax.set_ylabel("número de pares")
        ax.legend()
    fpr, tpr, _ = roc_curve(labels["test"].astype(int), scores["test"])
    axes[2].plot(fpr, tpr)
    axes[2].plot([0, 1], [0, 1], "k:", linewidth=1)
    axes[2].set_title("Curva ROC (teste)")
    axes[2].set_xlabel("FAR")
    axes[2].set_ylabel("1 - FRR")
    fig.tight_layout()
    fig.savefig(out_dir / "similarity_and_roc.png", dpi=130)
    plt.close(fig)


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

    # reprodutibilidade
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)

    from facerec.face.detection import FaceDetector
    from facerec.models.facenet import FaceNetEmbedder

    det_cfg, model_cfg = cfg["detector"], cfg["model"]
    detector = FaceDetector(det_cfg["min_confidence"], cfg["device"], det_cfg["min_face_size"])
    embedder = FaceNetEmbedder(model_cfg["pretrained"], cfg["device"])
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
    scores, labels, summary_counts, all_failures = {}, {}, {}, {}
    seconds_detect = seconds_embed = 0.0
    n_timed = 0

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
        seconds_detect += result.seconds_detect
        seconds_embed += result.seconds_embed
        n_timed += result.n_timed
        ok_items = [(p, i) for p, i in items if p in result.embeddings]
        pairs = make_verification_pairs(ok_items, n_pairs=cfg["pairs"][name], seed=seed + k)

        index = {p: n for n, p in enumerate(result.embeddings)}
        matrix = np.stack(list(result.embeddings.values()))
        idx_a = np.array([index[a] for a, _, _ in pairs])
        idx_b = np.array([index[b] for _, b, _ in pairs])
        labels[name] = np.array([y for _, _, y in pairs])
        scores[name] = SCORE_FUNCTIONS[cfg["score"]](matrix, idx_a, idx_b)
        summary_counts[name] = {
            "images": len(items),
            "embedded": len(ok_items),
            "failed": len(items) - len(ok_items),
            "identities": len({i for _, i in ok_items}),
            "pairs": len(pairs),
        }
        all_failures[name] = result.failures

    # 4. threshold escolhido SOMENTE na validação
    threshold = select_threshold(labels["val"], scores["val"], cfg["threshold_criterion"])
    val_metrics = evaluate_at_threshold(labels["val"], scores["val"], threshold)
    # 5. teste avaliado uma única vez, com o threshold da validação
    test_metrics = evaluate_at_threshold(labels["test"], scores["test"], threshold)

    plot_results(out_dir, scores, labels, threshold)
    failures_lines = [f"{n}\t{p}\t{r}" for n, d in all_failures.items() for p, r in sorted(d.items())]
    (out_dir / "detection_failures.txt").write_text("\n".join(failures_lines) + "\n", encoding="utf-8")

    report = {
        "experiment": cfg["experiment"],
        "limited_to_identities": args.limit_identities,
        "config": cfg,
        "versions": versions(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": cfg["device"],
        },
        "counts": summary_counts,
        "timing_seconds": {
            "images_processed": n_timed,
            "detect_total": round(seconds_detect, 2),
            "embed_total": round(seconds_embed, 2),
            "detect_per_image": round(seconds_detect / max(n_timed, 1), 4),
            "embed_per_image": round(seconds_embed / max(n_timed, 1), 4),
        },
        "threshold": {"criterion": cfg["threshold_criterion"], "value": threshold, "chosen_on": "val"},
        "val": val_metrics,
        "test": test_metrics,
        "test_eer_note": "O EER do teste é só uma medida da separação das distribuições; o threshold usado no teste é o da validação.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {cfg['experiment']} | score: {cfg['score']} | threshold (validação, {cfg['threshold_criterion']}): {threshold:.4f}")
    print(f"{'':10}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}{'AUC':>8}{'EER':>8}{'FAR':>8}{'FRR':>8}")
    for name, m in (("validação", val_metrics), ("teste", test_metrics)):
        print(
            f"{name:10}{m['accuracy']:8.4f}{m['precision']:8.4f}{m['recall']:8.4f}{m['f1']:8.4f}"
            f"{m['roc_auc']:8.4f}{m['eer']:8.4f}{m['far']:8.4f}{m['frr']:8.4f}"
        )
    print(f"imagens sem face detectada: {sum(c['failed'] for c in summary_counts.values())} de {sum(c['images'] for c in summary_counts.values())}")
    print(f"tempo por imagem: detecção {report['timing_seconds']['detect_per_image']}s, embedding {report['timing_seconds']['embed_per_image']}s")
    print(f"resultados em {out_dir}/")


if __name__ == "__main__":
    main()
