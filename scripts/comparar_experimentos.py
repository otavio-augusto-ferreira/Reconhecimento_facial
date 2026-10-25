"""Compara E01b, E02, E03... a partir de experiments/results/*/metrics.json (só dados da LFW).

Uso:  python scripts/comparar_experimentos.py
A DECISÃO do melhor modelo deve olhar as colunas de VALIDAÇÃO; o teste é só relato final (descrição, item 29).
"""
import json
from pathlib import Path

RES = Path("experiments/results")
OP = "far_0.1pct"


def op(rows):
    return next(r for r in rows if r["name"] == OP)


def line(name, epoch, d, params, secs):
    o = op(d["operating_points"])
    return (
        f"{name:14}{epoch:>7}{d['val_auc']:>9.4f}{d['val_eer']:>9.4f}{d['test_auc']:>9.4f}"
        f"{o['threshold']:>9.4f}{o['val']['far']:>9.4f}{o['val']['frr']:>9.4f}"
        f"{o['test']['far']:>9.4f}{o['test']['frr']:>9.4f}{o['test']['accuracy']:>9.4f}{params:>12}{secs:>9}"
    )


def main():
    head = (
        f"{'modelo':14}{'ép.':>7}{'valAUC':>9}{'valEER':>9}{'testAUC':>9}{'thr':>9}{'valFAR':>9}{'valFRR':>9}"
        f"{'tstFAR':>9}{'tstFRR':>9}{'tstAcc':>9}{'params':>12}{'treino_s':>9}"
    )
    print(f"Ponto de operação: {OP} (threshold escolhido na validação)\n" + head)
    base_printed = False
    for exp in sorted(p.name for p in RES.iterdir() if p.is_dir() and p.name.startswith("e0") and p.name not in ("e01", "e01b")):
        r = json.loads((RES / exp / "metrics.json").read_text(encoding="utf-8"))
        if not base_printed:
            print(line("pré-treinado", 0, r["baseline"], 0, 0.0))
            base_printed = True
        secs = round(sum(h["seconds"] for h in r["history"]), 0)
        print(line(f"{r['experiment']} {r['config']['train']['loss']}", r["best_epoch"], r["finetuned"],
                   r.get("trainable_parameters", "?"), secs))
    print("\nÉpoca 0 = modelo pré-treinado (a época é escolhida pelo EER da validação).")


if __name__ == "__main__":
    main()
