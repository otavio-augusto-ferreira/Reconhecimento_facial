"""Avaliação agregada no dataset PRIVADO. Rode só no seu computador; não envie fotos nem resultados por pessoa a ninguém.

Estrutura esperada (ignorada pelo Git):  data/private/<id_anonimo>/foto1.jpg, foto2.jpg, ...
Use ids anônimos (p01, p02, ...), nunca nomes. Cada pessoa precisa de pelo menos gallery_size + 1 fotos.

Uso (na raiz do repositório, com o ambiente ativo):
    python experiments/eval_private.py
Para avaliar outro modelo: --config configs/app_e03.yaml --output-dir experiments/results/private_e03.
O threshold vem de configs/app.yaml (escolhido na validação da LFW, ADR-003) e NÃO é ajustado com estes dados.
Sai só um resumo agregado em experiments/results/private/metrics.json (sem nomes, caminhos nem embeddings).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from facerec.app.config import load_config
from facerec.data.splits import scan_identity_folders
from facerec.pipeline import compute_embeddings
from facerec.private_eval import MIN_PERSONS, aggregate_verification, identification_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/private")
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--gallery-size", type=int, default=3, help="fotos de cada pessoa usadas como cadastro")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/results/private")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(args.root)
    items = [(p, i) for p, i in scan_identity_folders(root) if not Path(p).relative_to(root).parts[0].startswith("_")]
    if not items:
        raise SystemExit(f"nenhuma foto encontrada em {root}. Veja a estrutura esperada no início deste arquivo.")

    from facerec.face.detection import FaceDetector
    from facerec.models.facenet import FaceNetEmbedder

    detector = FaceDetector(cfg.detector_min_confidence, cfg.device, cfg.detector_min_face_size)
    embedder = FaceNetEmbedder(cfg.pretrained, cfg.device, checkpoint=cfg.checkpoint)
    signature = "|".join(str(v) for v in (cfg.pretrained, cfg.checkpoint, cfg.detector_min_confidence, cfg.detector_min_face_size, cfg.crop_margin_ratio, cfg.crop_rotate_to_eyes))
    res = compute_embeddings(
        [p for p, _ in items], detector, embedder,
        margin_ratio=cfg.crop_margin_ratio, rotate=cfg.crop_rotate_to_eyes,
        cache_path=root / "_cache" / "embeddings.npz", signature=signature,  # dentro de data/private: ignorado pelo Git
    )
    ok = [(p, i) for p, i in items if p in res.embeddings]
    emb = np.stack([res.embeddings[p] for p, _ in ok])
    identities = [i for _, i in ok]
    n_persons = len(set(identities))
    if n_persons < MIN_PERSONS:
        raise SystemExit(f"são necessárias pelo menos {MIN_PERSONS} pessoas com face detectada (há {n_persons}).")

    report = {
        "dataset": "privado (resultados agregados; nenhuma foto, nome ou embedding é guardado aqui)",
        "threshold": cfg.threshold,
        "threshold_origin": "validação da LFW (E01b, FAR <= 0,1%); não ajustado com dados privados",
        "counts": {
            "persons": n_persons,
            "images_total": len(items),
            "images_with_face": len(ok),
            "images_without_face": len(items) - len(ok),
        },
        "verification_pairs": aggregate_verification(emb, identities, cfg.threshold, args.seed),
        "identification_simulation": identification_simulation(emb, identities, args.gallery_size, cfg.threshold, args.seed),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    v, s = report["verification_pairs"], report["identification_simulation"]
    k, u = s["known_probes"], s["unknown_probes"]
    c = report["counts"]
    print(f"\n=== dataset privado (agregado) | threshold {cfg.threshold} (fixo, da LFW)")
    print(f"pessoas: {c['persons']} | fotos: {c['images_with_face']} com face, {c['images_without_face']} sem face")
    print(f"pares de verificação: {v['n_pairs']} | accuracy {v['accuracy']:.4f} | F1 {v['f1']:.4f} | AUC {v['roc_auc']:.4f} | FAR {v['far']:.4f} | FRR {v['frr']:.4f}")
    q = v["score_quantiles"]
    for nome, d in q.items():
        print(f"similaridade {nome:19}: min {d['min']:.3f} | p05 {d['p05']:.3f} | mediana {d['p50']:.3f} | p95 {d['p95']:.3f} | max {d['max']:.3f}")
    print(f"identificação (cadastro de {s['gallery_size']} fotos por pessoa):")
    print(f"  pessoas cadastradas: {k['correct']}/{k['n']} reconhecidas ({k['correct_rate']:.1%}), {k['wrong_person']} trocadas por outra pessoa, {k['rejected']} rejeitadas")
    print(f"  desconhecidos: {u['accepted']}/{u['n']} aceitos por engano ({u['false_accept_rate']:.1%})")
    print(f"\nresultado agregado em {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
