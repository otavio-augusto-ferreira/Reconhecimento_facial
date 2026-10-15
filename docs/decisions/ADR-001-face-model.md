# ADR-001 — Modelo de embedding facial

**Status**: pendente (decidir com os resultados da Parte 1)

## Contexto
Sem GPU local; fine-tuning só com dados públicos (LFW) em GPU gratuita de nuvem. A aplicação roda em CPU.

## Candidatos
- FaceNet (InceptionResnetV1, VGGFace2)
- ArcFace / MobileFaceNet (insightface)

## Critérios
Desempenho (EER, FAR/FRR, accuracy), custo computacional em CPU, licença dos pesos.

## Decisão
TODO após os experimentos E01–E06.
