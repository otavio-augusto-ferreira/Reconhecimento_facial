"""E02-E05 — fine-tuning do FaceNet (VGGFace2) com a divisão de TREINO da LFW (dados públicos).

Protocolo (docs/experiments/E02-finetune.md):
  1. detecta e recorta as faces de treino, validação e teste (cache em data/public/cache);
  2. mede o modelo pré-treinado (baseline) nos mesmos pares do E01b;
  3. treina; a cada época mede só a VALIDAÇÃO (AUC e EER) e guarda os pesos da melhor época;
  4. com os pesos da melhor época, escolhe os thresholds SÓ na validação e avalia o teste, uma vez.
O teste nunca decide nada: nem época, nem threshold, nem hiperparâmetro.

Uso (na raiz do repositório; veja docs/COLAB.md para rodar na GPU do Colab):
    python experiments/e02_finetune.py --config configs/e02_partial_ce.yaml
Teste rápido, com poucas pessoas:
    python experiments/e02_finetune.py --config configs/e02_partial_ce.yaml --limit-identities 20 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml
from e01_facenet_baseline import limit_identities, versions

from facerec.crops import compute_crops
from facerec.data.splits import load_splits, make_verification_pairs
from facerec.evaluation import format_operating_points, operating_points
from facerec.metrics import cosine_scores, roc_auc_eer


def pair_scores(emb: np.ndarray, paths: list[str], pairs):
    index = {p: i for i, p in enumerate(paths)}
    a = np.array([index[x] for x, _, _ in pairs])
    b = np.array([index[y] for _, y, _ in pairs])
    return np.array([lab for _, _, lab in pairs]), cosine_scores(emb, a, b)


def plot_history(out_dir: Path, history: list[dict], best_epoch: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    train = [(h["epoch"], h["train_loss"]) for h in history if h["train_loss"] is not None]
    ax1.plot(*zip(*train), marker="o") if train else None
    ax1.set_title("Perda de treino")
    ax1.set_xlabel("época")
    ax2.plot([h["epoch"] for h in history], [h["val_eer"] for h in history], marker="o", label="EER (validação)")
    ax2.axvline(best_epoch, color="k", linestyle="--", label=f"melhor época = {best_epoch}")
    ax2.set_title("EER na validação (época 0 = pré-treinado)")
    ax2.set_xlabel("época")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-identities", type=int, default=None, help="só para testes rápidos")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key, value in (("output_dir", args.output_dir), ("cache_dir", args.cache_dir), ("checkpoint_dir", args.checkpoint_dir)):
        if value:
            cfg[key] = value
    tcfg = cfg["train"]
    if args.epochs:
        tcfg["epochs"] = args.epochs
    seed = cfg["seed"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from facerec.face.detection import FaceDetector
    from facerec.training.data import limit_per_identity
    from facerec.training.losses import CosineClassifier
    from facerec.training.model import embed_crops, load_facenet, set_trainable
    from facerec.training.trainer import make_optimizer, train_one_epoch

    device = args.device or cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # reprodutibilidade na GPU (um pouco mais lento)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    print(f"dispositivo: {device}")

    # ------------------------------------------------------------ dados
    det_cfg = cfg["detector"]
    detector = FaceDetector(det_cfg["min_confidence"], device, det_cfg["min_face_size"])
    signature = "|".join(str(v) for v in (det_cfg["min_confidence"], det_cfg["min_face_size"], det_cfg["margin_ratio"], det_cfg["rotate_to_eyes"], 160))
    splits = load_splits(cfg["splits_manifest"], cfg["image_root"])
    items = {
        "train": limit_per_identity(splits["train"], cfg["data"]["max_images_per_identity"], seed),
        "val": splits["val"],
        "test": splits["test"],
    }
    if args.limit_identities:
        items = {k: limit_identities(v, args.limit_identities) for k, v in items.items()}
    crops, ok_items, failed = {}, {}, {}
    for name in ("train", "val", "test"):
        res = compute_crops(
            [p for p, _ in items[name]],
            detector,
            margin_ratio=det_cfg["margin_ratio"],
            rotate=det_cfg["rotate_to_eyes"],
            cache_path=Path(cfg["cache_dir"]) / f"crops_{name}{'_limited' if args.limit_identities else ''}.npz",
            signature=signature,
        )
        keep = set(res.paths)
        ok_items[name] = [(p, i) for p, i in items[name] if p in keep]
        crops[name] = res
        failed[name] = len(items[name]) - len(ok_items[name])
    pairs = {
        name: make_verification_pairs(ok_items[name], n_pairs=cfg["pairs"][name], seed=seed + k)
        for k, name in enumerate(("val", "test"))  # mesmas sementes do E01b: os pares são idênticos
    }
    identities = sorted({i for _, i in ok_items["train"]})
    ident_to_label = {i: n for n, i in enumerate(identities)}
    row = {p: n for n, p in enumerate(crops["train"].paths)}
    train_labels = np.array([ident_to_label[i] for p, i in sorted(ok_items["train"], key=lambda t: row[t[0]])])
    train_crops = crops["train"].crops[[row[p] for p, _ in sorted(ok_items["train"], key=lambda t: row[t[0]])]]
    counts = {
        "train": {"images": len(train_labels), "identities": len(identities), "failed": failed["train"]},
        "val": {"images": len(ok_items["val"]), "pairs": len(pairs["val"]), "failed": failed["val"]},
        "test": {"images": len(ok_items["test"]), "pairs": len(pairs["test"]), "failed": failed["test"]},
    }
    print("contagens:", json.dumps(counts))

    # ------------------------------------------------------------ baseline (modelo pré-treinado)
    model = load_facenet(cfg["model"]["pretrained"], device, dropout=0.0)

    def val_metrics() -> tuple[float, float]:
        emb = embed_crops(model, crops["val"].crops, device)
        y, s = pair_scores(emb, crops["val"].paths, pairs["val"])
        auc, eer, _ = roc_auc_eer(y, s)
        return auc, eer

    def full_report(tag: str) -> dict:
        labels, scores = {}, {}
        for name in ("val", "test"):
            emb = embed_crops(model, crops[name].crops, device)
            labels[name], scores[name] = pair_scores(emb, crops[name].paths, pairs[name])
        rows = operating_points(labels, scores, cfg["operating_points"], cfg["registered_people"])
        t_auc, t_eer, _ = roc_auc_eer(labels["test"], scores["test"])
        v_auc, v_eer, _ = roc_auc_eer(labels["val"], scores["val"])
        return {"tag": tag, "val_auc": v_auc, "val_eer": v_eer, "test_auc": t_auc, "test_eer_informative": t_eer, "operating_points": rows}

    baseline = full_report("pretrained")  # o teste entra só no relatório; nenhuma decisão depende dele
    auc0, eer0 = baseline["val_auc"], baseline["val_eer"]
    history = [{"epoch": 0, "train_loss": None, "val_auc": auc0, "val_eer": eer0, "seconds": 0.0}]
    print(f"época 0 (pré-treinado): val AUC {auc0:.4f}  val EER {eer0:.4f}")

    # ------------------------------------------------------------ treino
    n_trainable = set_trainable(model, tcfg["mode"])
    print(f"modo {tcfg['mode']}: {n_trainable:,} parâmetros treináveis | perda: {tcfg['loss']}")
    head = None
    if tcfg["loss"] in ("ce", "arcface"):
        margin = tcfg["margin"] if tcfg["loss"] == "arcface" else 0.0
        head = CosineClassifier(512, len(identities), tcfg["scale"], margin).to(device)
        if tcfg["head_init"] == "prototypes":
            emb0 = embed_crops(model, train_crops, device)
            head.init_from_prototypes(torch.from_numpy(emb0), torch.from_numpy(train_labels))
    optimizer = make_optimizer(model, head, tcfg["lr_backbone"], tcfg["lr_head"], tcfg["weight_decay"])
    if tcfg["loss"] == "triplet":
        steps = tcfg["steps_per_epoch"] or max(1, len(train_labels) // (tcfg["p"] * tcfg["k"]))
    else:
        steps = len(train_labels) // tcfg["batch_size"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps * tcfg["epochs"]))

    better = (lambda new, old: new < old) if tcfg["select_metric"] == "eer" else (lambda new, old: new > old)
    key = "val_eer" if tcfg["select_metric"] == "eer" else "val_auc"
    best_epoch, best_value = 0, history[0][key]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(
            model, head, tcfg["loss"], train_crops, train_labels,
            mode=tcfg["mode"], optimizer=optimizer, scheduler=scheduler,
            batch_size=tcfg["batch_size"], p=tcfg["p"], k=tcfg["k"],
            steps=tcfg["steps_per_epoch"], triplet_margin=tcfg["triplet_margin"],
            use_augment=tcfg["augment"], rng=rng, gen=gen, device=device,
        )
        auc, eer = val_metrics()
        history.append({"epoch": epoch, "train_loss": loss, "val_auc": auc, "val_eer": eer, "seconds": round(time.perf_counter() - t0, 1)})
        mark = ""
        if better(history[-1][key], best_value):
            best_epoch, best_value = epoch, history[-1][key]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- melhor"
        print(f"época {epoch:2d}: perda {loss:.4f}  val AUC {auc:.4f}  val EER {eer:.4f}  ({history[-1]['seconds']} s){mark}")

    # ------------------------------------------------------------ avaliação final (teste uma única vez)
    model.load_state_dict(best_state)
    finetuned = full_report(f"{cfg['experiment']} (época {best_epoch})")

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['experiment'].lower()}{'_limited' if args.limit_identities else ''}.pt"
    torch.save({"state_dict": best_state, "experiment": cfg["experiment"], "best_epoch": best_epoch, "config": cfg}, ckpt_path)
    plot_history(out_dir, history, best_epoch)

    report = {
        "experiment": cfg["experiment"],
        "limited_to_identities": args.limit_identities,
        "config": cfg,
        "versions": {**versions(), "device": device},
        "counts": counts,
        "trainable_parameters": n_trainable,
        "history": history,
        "best_epoch": best_epoch,
        "baseline": baseline,
        "finetuned": finetuned,
        "note": "Época e thresholds escolhidos só na validação; o teste entra apenas no relatório final. O checkpoint (.pt) não vai ao Git.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    n_reg = cfg["registered_people"]
    print(f"\n=== {cfg['experiment']} | modo {tcfg['mode']} | perda {tcfg['loss']} | melhor época: {best_epoch} (critério: {key})")
    print(f"\n--- baseline (FaceNet pré-treinado, mesmos pares) | teste AUC {baseline['test_auc']:.4f}")
    print(format_operating_points(baseline["operating_points"], n_reg))
    print(f"\n--- {finetuned['tag']} | teste AUC {finetuned['test_auc']:.4f}")
    print(format_operating_points(finetuned["operating_points"], n_reg))
    print(f"\nresultados em {out_dir}/ | checkpoint (não versionar): {ckpt_path}")


if __name__ == "__main__":
    main()
