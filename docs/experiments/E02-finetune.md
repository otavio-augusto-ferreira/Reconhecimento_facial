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
Executados: **E02 e E03** (GPU T4, Colab). **E04 (triplet) e E05 (rede inteira) não foram executados** por falta de tempo; as hipóteses H2 (parte do triplet) e H3 ficam sem teste.

Todos os valores usam os mesmos pares do E01b, threshold FAR ≤ 0,1% escolhido na validação de cada modelo, e treino de 10 épocas (≈ 94 s por experimento na T4; 10 523 136 parâmetros treináveis).

| Id | Melhor época | Val AUC | Val EER | Threshold | Teste AUC | Teste FAR | Teste FRR | Teste acc | Teste F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pré-treinado (época 0) | 0 | 0,9828 | 0,0507 | 0,5404 | 0,9697 | 0,0017 | 0,1570 | 0,9207 | 0,9140 |
| E02 (softmax de cosseno) | 0 | 0,9828 | 0,0507 | 0,5404 | 0,9697 | 0,0017 | 0,1570 | 0,9207 | 0,9140 |
| E03 (ArcFace) | 8 | 0,9858 | 0,0503 | 0,4609 | 0,9714 | 0,0031 | 0,1357 | 0,9306 | 0,9257 |
| E04 (triplet) | não executado | | | | | | | | |
| E05 (rede inteira) | não executado | | | | | | | | |

O baseline do E01b (0,9215 de acurácia no teste; 7 falsos aceites em 4172) difere por 0,0008 do pré-treinado acima. Os dois usam os mesmos pares, mas o E02/E03 usa o pipeline com recortes em cache e GPU; a causa exata da pequena diferença não foi verificada.

## Interpretação
- **H1 (CE melhora a validação): refutada.** Nenhuma época de E02 superou o modelo pré-treinado; o resultado é idêntico ao da época 0.
- **H2 (ArcFace/triplet reduzem o FRR com FAR ≤ 0,1% em relação ao E02): confirmada para o ArcFace, não testada para o triplet.** O FRR do ArcFace foi 0,0918 contra 0,0995 na validação e 0,1357 contra 0,1570 no teste.
- **H3 (rede inteira não supera o ajuste parcial): não testada.**
- O ganho do E03 é pequeno (EER de validação 0,0507 para 0,0503); o FAR no teste foi maior (≈ 13 contra 7 falsos aceites em 4172 pares), diferença que o acaso pode explicar com tão poucos eventos (não testado formalmente).
- Decisão sobre o modelo da aplicação: ver `docs/decisions/ADR-001-face-model.md`. Análise completa: `docs/ANALISE.md`.
