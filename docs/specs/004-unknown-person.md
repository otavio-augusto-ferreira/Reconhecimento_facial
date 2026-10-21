# 004 — Pessoa desconhecida

**Status**: implementada

## Problema
O sistema não deve forçar toda face a uma identidade conhecida.

## Requisitos
- R1: se similaridade máxima < threshold → "desconhecido".
- R2: o threshold vem de configuração externa e foi escolhido no conjunto de validação (nunca no teste).
- R3: registrar FAR e FRR no threshold escolhido (ver ADR-003).

## Critérios de aceitação
- CA1: face de pessoa não cadastrada → "desconhecido".
- CA2: FAR/FRR no teste reportados no README sem identificar participantes.

## Implementação
O threshold vem de `configs/app.yaml` (`recognition.threshold`). Valor atual: 0,5404, escolhido na validação do E01b como o menor FRR com FAR ≤ 0,1% (ADR-003). Aceita-se quando a similaridade é maior ou igual ao threshold.
