"""Atualiza o README.md com os resultados que existem em experiments/results/ e as seções que ainda são TODO.

Uso (na raiz do repositório):
    python scripts/update_readme.py
Depois confira com `git diff README.md`. É idempotente: pode rodar de novo quando houver novos resultados (E02-E05).
Só lê arquivos de resultados agregados (experiments/results/*/metrics.json); nunca dados privados.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

README = Path("README.md")
RES = Path("experiments/results")
MODEL = "FaceNet (InceptionResnetV1, VGGFace2)"
OP = "far_0.1pct"  # ponto de operação adotado na aplicação (ADR-003)


def f(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def load(name: str):
    p = RES / name / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def op_row(rows: list[dict], name: str):
    return next((r for r in rows if r["name"] == name), None)


def results_rows() -> list[str]:
    out = []
    e01 = load("e01")
    if e01:
        t = e01["test"]
        out.append(
            f"| E01 | {MODEL} | pré-treinado, sem fine-tuning; threshold por acurácia na validação | público (LFW) | "
            f"{f(t['accuracy'])} | {f(t['f1'])} | Teste, cosseno, threshold {f(t['threshold'])}. AUC {f(t['roc_auc'])}, "
            f"EER {f(t['eer'])}, FAR {f(t['far'])}, FRR {f(t['frr'])} |"
        )
    e01b = load("e01b")
    if e01b and (r := op_row(e01b["operating_points"], OP)):
        t = r["test"]
        out.append(
            f"| E01b | {MODEL} | pré-treinado; threshold por FAR ≤ 0,1% na validação (adotado na aplicação) | público (LFW) | "
            f"{f(t['accuracy'])} | {f(t['f1'])} | Teste, threshold {f(r['threshold'])}. FAR {f(t['far'])}, FRR {f(t['frr'])} "
            f"({t['confusion']['fp']} falsos aceites em {t['confusion']['fp'] + t['confusion']['tn']} pares negativos) |"
        )
    for exp in ("e02", "e03", "e04", "e05"):
        rep = load(exp)
        if not rep:
            continue
        ft = rep["finetuned"]
        r = op_row(ft["operating_points"], OP)
        tr = rep["config"]["train"]
        strategy = f"fine-tuning {'parcial (último estágio)' if tr['mode'] == 'last_block' else 'completo'}, perda {tr['loss']}, melhor época {rep['best_epoch']}"
        note = "melhor época 0 = igual ao pré-treinado (o ajuste não melhorou a validação). " if rep["best_epoch"] == 0 else ""
        t = r["test"]
        out.append(
            f"| {rep['experiment']} | {MODEL} | {strategy} | público (LFW, treino) | {f(t['accuracy'])} | {f(t['f1'])} | "
            f"{note}Teste, threshold FAR ≤ 0,1% da validação ({f(r['threshold'])}). AUC {f(ft['test_auc'])}, FAR {f(t['far'])}, FRR {f(t['frr'])} |"
        )
    return out


def private_paragraph() -> str | None:
    rep = load("private")
    if not rep:
        return None
    v, s, c = rep["verification_pairs"], rep["identification_simulation"], rep["counts"]
    k, u = s["known_probes"], s["unknown_probes"]
    return (
        f"Dataset privado (apenas adultos com consentimento; resultados agregados, threshold {rep['threshold']} fixo da LFW): "
        f"{c['persons']} pessoas e {c['images_with_face']} fotos. Verificação em {v['n_pairs']} pares: accuracy {f(v['accuracy'])}, "
        f"F1 {f(v['f1'])}, AUC {f(v['roc_auc'])}, FAR {f(v['far'])}, FRR {f(v['frr'])}. Identificação com cadastro de {s['gallery_size']} fotos por pessoa: "
        f"{k['correct_rate']:.1%} das tentativas de pessoas cadastradas reconhecidas corretamente e {u['false_accept_rate']:.1%} dos desconhecidos aceitos por engano."
    )


MODELS_SECTION = """## Modelos pré-treinados

| Modelo | Fonte | Pesos | Licença / termos | Uso no projeto |
|---|---|---|---|---|
| FaceNet (InceptionResnetV1) | pacote `facenet-pytorch` (https://github.com/timesler/facenet-pytorch) | Treinados na VGGFace2 (o README do pacote cita também a CASIA-Webface); baixados automaticamente na primeira execução (cerca de 107 MB) e não versionados | Código do pacote: MIT (verificado na página do repositório em 25/10/2026). O README do pacote não informa a licença dos pesos nem os termos da VGGFace2, e não localizei termos oficiais da VGGFace2; usamos apenas os pesos prontos, sem baixar nem redistribuir a base. Pendente: confirmar com o professor | Embeddings de 512 dimensões: baseline e ponto de partida do fine-tuning |
| MTCNN | pacote `facenet-pytorch` | Incluídos no pacote; o README não informa a origem nem a licença dos pesos | Código do pacote: MIT. Licença dos pesos: não informada na fonte | Detecção de faces e pontos de referência |

Nota: o pré-treinamento na VGGFace2 pode ter identidades em comum com a LFW, o que pode tornar os resultados otimistas (limitação registrada).
"""

REPRO_SECTION = """Veja `docs/EXECUCAO.md` (ambiente, experimentos e aplicação) e `docs/COLAB.md` (fine-tuning na GPU do Colab). Resumo:

```
pip install -r requirements.txt && pytest
python experiments/e01_facenet_baseline.py --config configs/e01_facenet_baseline.yaml
python experiments/e01b_operating_points.py --config configs/e01b_operating_points.yaml
python experiments/e02_finetune.py --config configs/e02_partial_ce.yaml      # GPU (Colab)
python -m uvicorn facerec.app.main:app --host 127.0.0.1 --port 8000          # aplicação em http://127.0.0.1:8000
```

O dataset público é obtido localmente (`data/README.md`). Nenhum dado privado faz parte do repositório."""


def main() -> int:
    if not README.exists():
        print("README.md não encontrado: rode na raiz do repositório.")
        return 1
    text = README.read_text(encoding="utf-8")
    lines = text.split("\n")
    changes = []

    # tabela de resultados: substitui as linhas de dados (mantém o cabeçalho)
    rows = results_rows()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("| Experimento |")), None)
    if start is not None and rows:
        end = start + 2
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        lines[start + 2 : end] = rows
        changes.append(f"tabela de resultados ({len(rows)} linhas)")
    text = "\n".join(lines)

    def sub(pattern: str, repl: str, label: str, flags=0):
        nonlocal text
        new, n = re.subn(pattern, lambda _: repl, text, count=1, flags=flags)
        if n:
            text = new
            changes.append(label)

    sub(r"\(`src/app/`\)", "(`src/facerec/app/`)", "caminho da Parte 2")
    sub(r"## Modelos pré-treinados\n+TODO: listar modelo, fonte e licença\.\n?", MODELS_SECTION, "modelos pré-treinados")
    sub(r"TODO: instruções de ambiente, download dos datasets e execução dos experimentos\.", REPRO_SECTION, "reprodução")

    para = private_paragraph()
    marker = "Resultados com o dataset privado são apresentados apenas de forma agregada."
    if para and marker in text and para not in text:
        old = re.search(r"Dataset privado \(apenas adultos.*\n?", text)
        text = text.replace(old.group(0), "") if old else text
        text = text.replace(marker, marker + "\n\n" + para)
        changes.append("resultado agregado do dataset privado")

    README.write_text(text, encoding="utf-8")
    print("atualizado:", ", ".join(changes) if changes else "nada para alterar (já está em dia)")
    print("confira com: git diff README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
