# 003 — Reconhecimento facial

**Status**: implementada

## Problema
Comparar o embedding de uma face com os cadastrados e decidir quem é.

## Requisitos
- R1: gerar embedding com o modelo escolhido na Parte 1 (ver ADR-001).
- R2: comparar com os embeddings cadastrados usando a métrica definida na Parte 1 (cosseno ou euclidiana).
- R3: retornar a identidade mais similar e o score.
- R4: aplicar threshold (ver 004 e ADR-003).

## Critérios de aceitação
- CA1: pessoa cadastrada, imagem nova → identidade correta com score acima do threshold.
- CA2: tempo de inferência por face medido e registrado (CPU).
- CA3: banco vazio → resultado "desconhecido", sem erro.

## Implementação
`FaceService.recognize`: detecta todas as faces da imagem, gera um embedding por face e compara com os embeddings cadastrados por similaridade de cosseno (produto escalar de vetores normalizados). A pessoa é a de maior similaridade entre as suas imagens cadastradas. A resposta traz o tempo de detecção e de embedding em ms (CA2). Testes: `tests/test_app_service.py`.
