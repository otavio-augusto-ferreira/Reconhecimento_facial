# 001 — Detecção e alinhamento de faces

**Status**: rascunho

## Problema
Dada uma imagem, localizar faces e produzir um recorte alinhado para o modelo de embedding.

## Requisitos
- R1: detectar faces em imagem estática e frame de câmera.
- R2: retornar bounding box, landmarks e confiança.
- R3: alinhar e redimensionar o recorte para o tamanho de entrada do modelo.
- R4: tratar zero faces e múltiplas faces como resultados explícitos.

## Critérios de aceitação
- CA1: imagem com uma face → um recorte alinhado.
- CA2: imagem sem face → erro `NoFaceDetected`, sem exceção não tratada.
- CA3: imagem com várias faces → lista ordenada por confiança.
- CA4: confiança mínima configurável em `configs/`.

## Fora de escopo
Detecção de vivacidade (liveness).
