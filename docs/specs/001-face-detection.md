# 001 — Detecção e alinhamento de faces

**Status**: implementada em `src/facerec/face/detection.py`; testes em `tests/test_detection.py`.

## Problema
Dada uma imagem, localizar faces e produzir um recorte alinhado para o modelo de embedding.

## Requisitos
- R1: detectar faces em imagem estática e frame de câmera.
- R2: retornar bounding box, landmarks e confiança.
- R3: alinhar e redimensionar o recorte para o tamanho de entrada do modelo (160 x 160).
- R4: tratar zero faces e múltiplas faces como resultados explícitos.
- R5: a margem do recorte e a rotação pelos olhos são configuráveis (`margin_ratio`, `rotate_to_eyes`), para serem tratadas como variáveis experimentais.

## Critérios de aceitação
- CA1: imagem com uma face → um recorte alinhado.
- CA2: imagem sem face → erro `NoFaceDetected`, sem exceção não tratada (`test_blank_image_has_no_face`).
- CA3: imagem com várias faces → lista ordenada por confiança (`FaceDetector.detect`).
- CA4: confiança mínima configurável em `configs/` (`detector.min_confidence`).

## Decisões
- Detector: MTCNN do `facenet-pytorch`, na CPU.
- No pipeline dos experimentos, imagens sem face são excluídas da avaliação e contadas, em vez de receberem um recorte de reserva. Assim a taxa de falha da detecção aparece nos resultados.
- A rotação pelos olhos gira a imagem em torno do ponto médio dos olhos; o teste sintético confere que os olhos ficam na mesma altura depois do alinhamento.

## Fora de escopo
Detecção de vivacidade (liveness).
