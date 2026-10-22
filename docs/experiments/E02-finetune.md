# E02–E05 — Fine-tuning do FaceNet com a LFW

## Pergunta
Ajustar o FaceNet pré-treinado (VGGFace2) com as identidades de **treino** da LFW melhora a verificação em identidades **novas** (validação e teste)? Qual estratégia de ajuste e qual função de perda funcionam melhor?

## Hipóteses
- **H1**: o fine-tuning parcial com softmax de cosseno (E02) reduz o EER de validação e aumenta o ROC-AUC de teste em relação ao modelo pré-treinado, nos mesmos pares.
- **H2**: ArcFace (E03) ou triplet loss (E04) reduzem o FRR no ponto `FAR ≤ 0,1%` em relação ao softmax de cosseno (E02).
- **H3**: o ajuste da rede inteira (E05), com cerca de 5 mil imagens de treino, não supera o ajuste parcial (E02/E03) por sobreajuste.

Expectativa honesta: o modelo pré-treinado já é forte (E01: AUC de teste 0,966) e foi treinado com muito mais dados. Ganhos pequenos ou nulos são plausíveis, e "nenhuma época melhorou a validação" é um resultado válido.

## Experimentos
| Id | Config | O que treina | Perda | Learning rate (rede) |
| --- | --- | --- | --- | --- |
| E02 | `configs/e02_partial_ce.yaml` | último estágio (`repeat_3`, `block8`, `last_linear`, `last_bn`) | softmax de cosseno (escala 30) | 5e-5 |
| E03 | `configs/e03_partial_arcface.yaml` | idem | ArcFace (escala 30, margem 0,3 rad) | 5e-5 |
| E04 | `configs/e04_partial_triplet.yaml` | idem | triplet loss batch-hard (margem 0,2; lotes 16 × 4) | 5e-5 |
| E05 | `configs/e05_full_arcface.yaml` | rede inteira | ArcFace (escala 30, margem 0,3 rad) | 1e-5 |

Comum a todos: seed 42; no máximo 20 imagens por identidade no treino (para que pessoas muito fotografadas não dominem); aumento de dados leve (espelhamento, brilho, contraste e translação de até 8 px); AdamW com decaimento cosseno; 10 épocas; dropout desligado; o classificador de E02/E03 começa nos **protótipos** (média dos embeddings de cada identidade), para o treino partir de um ponto já bom e não destruir o que o modelo aprendeu; as camadas congeladas ficam em modo `eval` (o BatchNorm delas não muda).

## Protocolo (sem vazamento)
1. Treino usa **só** a divisão `train` do manifesto `experiments/splits/lfw_seed42.json` (1176 identidades). Validação e teste têm identidades que o modelo nunca viu no treino.
2. Os pares de validação e teste são **idênticos** aos do E01b (mesma função e mesmas sementes), então os números são comparáveis linha a linha.
3. A cada época mede-se **só a validação** (AUC e EER). Os pesos da melhor época (EER mínimo na validação; a época 0, pré-treinado, também concorre) são guardados.
4. Com esses pesos, os thresholds dos quatro pontos de operação (acurácia máxima, EER, FAR ≤ 1%, FAR ≤ 0,1%) são escolhidos **só na validação** e aplicados ao teste, uma única vez.
5. O baseline (modelo pré-treinado) é medido no teste para relatório, mas nenhuma decisão depende dele.

## Como reproduzir
Veja `docs/COLAB.md` (GPU) e `docs/EXECUCAO.md`.

```
python experiments/e02_finetune.py --config configs/e02_partial_ce.yaml
```

Saída: `experiments/results/<id>/metrics.json` (histórico por época, as duas tabelas, contagens, versões) e `training_curves.png`. Os pesos ficam em `checkpoints/public/` (não vão ao Git).

## Cuidados metodológicos
- O teste não escolhe época, threshold nem hiperparâmetro. Se você mudar um hiperparâmetro depois de ver o teste, o resultado deixa de valer: registre a mudança e o motivo.
- A VGGFace2 (pré-treino) pode ter identidades em comum com a LFW. Isso pode inflar tanto o baseline quanto o fine-tuning e deve ser declarado como limitação.
- Os pesos treinados são derivados de dados públicos (LFW) e de pesos de terceiros: confira a licença antes de redistribuir. Não vão ao Git.
- A seleção de época usa o EER da validação com 17 562 pares; diferenças pequenas entre épocas estão dentro do ruído. Prefira conclusões baseadas na tabela de pontos de operação e em todos os experimentos juntos.
- Resultados na LFW não valem para outros grupos (ver o aviso oficial no README) nem para fotos de câmera de baixa qualidade.

## Resultado
A preencher depois de rodar cada experimento (copie da saída do script).

| Id | Melhor época | Teste AUC | Threshold (FAR ≤ 0,1%, val) | Teste FAR | Teste FRR | Falsos aceites |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (E01b) | — | 0,9657* | 0,5404 | 0,0017 | 0,1553 | 7 de 4172 |
| E02 | | | | | | |
| E03 | | | | | | |
| E04 | | | | | | |
| E05 | | | | | | |

\* AUC de teste do E01 (3 000 pares). Use o AUC do baseline impresso pelo próprio script, calculado nos mesmos pares do fine-tuning.

## Interpretação
A preencher: H1, H2 e H3 foram confirmadas? A melhor estratégia melhora o ponto `FAR ≤ 0,1%` que a aplicação usa? Vale trocar o modelo da aplicação (ADR-001)?
