# E01 — FaceNet pré-treinado (VGGFace2), sem fine-tuning

**Status**: código pronto e testado; execução no LFW completo pendente (preencher "Resultado" e "Interpretação").

## Pergunta
Qual o desempenho de um modelo de embeddings faciais pré-treinado, sem nenhum ajuste ao nosso conjunto, na verificação de pares do LFW? Este é o ponto de partida (baseline) contra o qual os fine-tunings (E02 em diante) serão comparados.

## Hipóteses
Definidas antes de ver os resultados no teste.

- **H1.** O InceptionResnetV1 pré-treinado em VGGFace2, com similaridade de cosseno, separa pares da mesma pessoa de pares de pessoas diferentes no LFW com ROC-AUC acima de 0,95.
- **H2.** O threshold escolhido na validação generaliza para o teste: FAR e FRR no teste diferem dos da validação em menos de 0,02 (valor absoluto).

## Configuração

| Item | Valor |
| --- | --- |
| Dataset | LFW, divisão por identidade (seed 42): 1.176 / 252 / 252 identidades em treino / validação / teste |
| Conjuntos usados | Validação (escolher o threshold) e teste (avaliar uma vez). O treino não é usado: não há treinamento |
| Modelo | InceptionResnetV1 (`facenet-pytorch` 2.6.0), pesos `vggface2`, embedding de 512 dimensões |
| Detecção | MTCNN, confiança mínima 0,9, face mínima de 20 px, recorte sem margem, sem rotação (`configs/e01_facenet_baseline.yaml`) |
| Estratégia de treinamento | Nenhuma (modelo congelado, só inferência) |
| Comparação | Similaridade de cosseno |
| Pares | 3.000 em cada conjunto, 50% mesma pessoa e 50% pessoas diferentes (podem ser menos se não houver pares positivos distintos suficientes) |
| Threshold | Escolhido na validação, critério acurácia máxima; aceito se score ≥ threshold |
| Hiperparâmetros de inferência | Lote de 32 imagens, CPU, seed 42 |
| Métricas | Accuracy, precision, recall, F1, ROC-AUC, EER, FAR, FRR, matriz de confusão, distribuição das similaridades |

## Como reproduzir
Com o ambiente criado (`requirements.txt`) e o LFW em `data/public/lfw/` (ver `data/README.md`):

```
python experiments/make_lfw_splits.py --root data/public/lfw
python experiments/e01_facenet_baseline.py --config configs/e01_facenet_baseline.yaml --limit-identities 20
python experiments/e01_facenet_baseline.py --config configs/e01_facenet_baseline.yaml
```

O segundo comando é um teste rápido com 20 identidades por conjunto. Na primeira execução o PyTorch baixa os pesos (cerca de 107 MB). Os embeddings ficam em cache em `data/public/cache/` (ignorado pelo Git), então rodar de novo só com outro threshold ou outras métricas é rápido. Saídas em `experiments/results/e01/`: `metrics.json`, `similarity_and_roc.png` e `detection_failures.txt`. Todas contêm apenas dados do dataset público.

## Cuidados metodológicos
- **O teste não escolhe nada.** O threshold vem da validação. Variações futuras (margem do recorte, rotação pelos olhos, métrica de comparação) devem ser comparadas na validação; se a configuração mudar depois de olhar o teste, o resultado vira um novo experimento e não substitui este.
- **Imagens sem face detectada são excluídas da avaliação e contadas** (`counts.failed` no `metrics.json`; lista em `detection_failures.txt`). Uma taxa de falha alta enviesaria o resultado a favor das imagens fáceis e deve ser discutida na análise.
- **Possível sobreposição com o treino do modelo.** Os pesos foram treinados em VGGFace2. Não verificamos se há identidades do LFW nesse conjunto de treino; isso precisa ser checado na documentação do modelo e registrado como limitação.
- **Licenças.** Registrar no README a licença dos pesos pré-treinados e a do dataset em que foram treinados.

## Resultado
Execução completa em 20/10/2026, LFW inteiro, seed 42, CPU, score cosseno.

| Conjunto | Accuracy | Precision | Recall | F1 | ROC-AUC | EER | FAR | FRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validação | 0,9643 | 0,9895 | 0,9387 | 0,9634 | 0,9778 | 0,0500 | 0,0100 | 0,0613 |
| Teste | 0,9477 | 0,9891 | 0,9053 | 0,9454 | 0,9657 | 0,0733 | 0,0100 | 0,0947 |

Threshold: 0,4009 (acurácia máxima na validação, aplicado sem alteração no teste). Imagens sem face detectada: 0 de 2278. Tempo por imagem na CPU: detecção 0,0612 s, embedding 0,0351 s.

Observação: o teste rápido com 20 identidades (258 imagens) deu AUC 0,9517 no teste; era só verificação do fluxo e não entra nos resultados.

## Interpretação
- **H1 (ROC-AUC > 0,95 com cosseno): confirmada.** AUC de 0,9657 no teste e 0,9778 na validação.
- **H2 (FAR e FRR do teste a até 0,02 dos da validação): parcialmente rejeitada.** O FAR ficou idêntico (0,0100 nos dois). O FRR subiu de 0,0613 para 0,0947, diferença de 0,0334, acima da tolerância. Com cerca de 1.500 pares positivos, o erro-padrão do FRR é de aproximadamente 0,007, então a diferença é maior do que o esperado só por sorteio de pares.
- **Tipo de erro predominante:** falsos rejeites. O critério de acurácia máxima escolheu um ponto de operação conservador (FAR 1%, FRR 9,5% no teste), em que o sistema prefere rejeitar a aceitar a pessoa errada. Isso é coerente com o contexto de reconhecimento, em que aceitar um desconhecido costuma ser mais grave. Mas o EER do teste (0,0733) mostra que existe um ponto com erros mais equilibrados.
- **Generalização:** a queda de validação para teste (AUC 0,9778 → 0,9657; EER 0,0500 → 0,0733) é pequena e esperada, já que as identidades de teste não foram vistas nem na escolha do threshold. O que não dá para afirmar ainda é quanto desse desempenho vem de o modelo ter sido pré-treinado na VGGFace2, que pode ter identidades em comum com a LFW. Isso entra como limitação.
- **Detecção:** 0 falhas em 2.278 imagens; a LFW tem rostos frontais e bem enquadrados, então essa taxa não deve ser esperada em fotos tiradas por webcam ou celular (Parte 2).
- **Custo:** cerca de 0,096 s por imagem (detecção + embedding) na CPU, ou 10 imagens por segundo, suficiente para uso interativo na aplicação.
- **Próximos passos derivados:** (1) rodar o mesmo modelo com critério `eer` para o threshold, como comparação do ponto de operação; (2) usar este resultado como baseline de E02 em diante, com o fine-tuning comparado nas mesmas divisões e nos mesmos pares.
