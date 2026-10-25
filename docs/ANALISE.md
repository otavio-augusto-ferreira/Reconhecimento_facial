# Análise dos resultados (Parte 1)

Todos os números de LFW usam a divisão por identidade com seed 42 (1176 / 252 / 252 identidades), os mesmos pares em todos os experimentos e thresholds escolhidos só na validação. Resultados do dataset privado são agregados (3 participantes adultos com consentimento; 13 fotos).

## 1. Qual modelo foi melhor e por quê
- **Pelo protocolo (validação), o E03: FaceNet com ajuste parcial e perda ArcFace.** Val AUC 0,9858 (contra 0,9828), val EER 0,0503 (contra 0,0507) e FRR de validação 0,0918 (contra 0,0995) com FAR ≤ 0,1%. No teste: acurácia 0,9306 (contra 0,9207), F1 0,9257 (contra 0,9140), FRR 0,1357 (contra 0,1570).
- O ganho é **pequeno**. O modelo pré-treinado já é forte: foi treinado em um conjunto muito maior (VGGFace2), e o ajuste usa só 1176 identidades de treino da LFW.
- Plausível, mas não demonstrado: a margem angular do ArcFace separa melhor as classes no espaço de embeddings do que o softmax de cosseno com margem 0, e por isso só o E03 melhorou a validação.

## 2. Modificações que melhoraram e que pioraram
| Modificação | Efeito |
| --- | --- |
| E02: softmax de cosseno, último estágio | Nenhuma época superou o pré-treinado (melhor época = 0). Sem ganho. |
| E03: ArcFace (escala 30, margem 0,3), último estágio | Pequena melhora de AUC e FRR; FAR de teste maior (0,0031 contra 0,0017). |
| Threshold por FAR ≤ 0,1% (E01b) em vez de acurácia máxima (E01) | FAR de teste de 0,0100 para 0,0017; FRR de 0,0947 para 0,1553. Troca deliberada (ADR-003). |
| E04 (triplet) e E05 (rede inteira) | Não executados. |

## 3. Impacto do fine-tuning
Parcial e modesto: só a perda ArcFace melhorou a validação. O fine-tuning com dados limitados corre o risco de sobreajustar às identidades de treino, e o critério "melhor época pela validação, com a época 0 como candidata" impede piorar o modelo sem perceber (foi o que preveniu o E02 de ser pior que o baseline).

## 4. Impacto do threshold (a maior alavanca)
- Com o threshold que maximiza a acurácia (E01, 0,4009), o FAR de teste foi 0,0100; com FAR ≤ 0,1% na validação (0,5404), 0,0017. O custo foi o FRR de 0,0947 para 0,1553.
- O risco de aceitar um desconhecido cresce com o número de pessoas cadastradas, aproximadamente 1 − (1 − FAR)^N (E01b). Por isso a aplicação usa o ponto mais rígido.
- O threshold **depende do modelo**: o E03 escolhe 0,4609 para o mesmo critério. Thresholds não são comparáveis entre modelos.

## 5. Distribuição de similaridades (mesma pessoa contra pessoas diferentes)
Para a LFW, `experiments/results/e01/similarity_and_roc.png` mostra os histogramas e a curva ROC.

No dataset privado (23 pares de cada classe; valores agregados em `experiments/results/private*/metrics.json`):

| Modelo | Classe | Mín. | p05 | Mediana | p95 | Máx. |
| --- | --- | --- | --- | --- | --- | --- |
| Pré-treinado | mesma pessoa | 0,576 | 0,624 | 0,857 | 0,928 | 0,969 |
| Pré-treinado | pessoas diferentes | 0,264 | 0,299 | 0,444 | 0,580 | 0,605 |
| E03 | mesma pessoa | 0,594 | 0,633 | 0,881 | 0,932 | 0,971 |
| E03 | pessoas diferentes | 0,372 | 0,377 | 0,472 | 0,626 | 0,641 |

Threshold do pré-treinado: 0,5404. O do E03: 0,4609.

## 6. Erros encontrados
- **No dataset privado o threshold da LFW foi permissivo demais.** Com o modelo pré-treinado e 0,5404: AUC 0,9962, FRR 0,0000, mas FAR 0,2174 nos pares e 4 de 7 desconhecidos simulados aceitos. Com o E03 e 0,4609: AUC 0,9943, FAR 0,6522 e 7 de 7 desconhecidos aceitos. Em ambos, as pessoas cadastradas foram reconhecidas (7 de 7) e nenhuma foi trocada por outra.
- O AUC alto indica que a **ordenação** das similaridades é boa e que o problema é a **calibração do threshold** para esse domínio. A tabela da seção 5 mostra como:
  - As similaridades de **mesma pessoa** no dataset privado são altas (mediana 0,86 a 0,88; mínimo 0,58 a 0,59), bem acima dos dois thresholds. Fotos da mesma pessoa tiradas em condições parecidas ficam muito próximas; na LFW, os pares positivos vêm de situações diferentes, o que empurra o threshold para baixo. É uma hipótese coerente com os dados, não uma demonstração.
  - As similaridades entre **pessoas diferentes** também são altas: no pré-treinado, o percentil 95 é 0,580 e o máximo 0,605, acima do threshold 0,5404; por isso 22% dos pares negativos foram aceitos. As faixas dos dois grupos só se sobrepõem entre cerca de 0,576 e 0,605.
  - No E03, a distribuição de pessoas diferentes se deslocou para cima (mediana 0,472; mínimo 0,372) e o threshold dele (0,4609) ficou abaixo da mediana dos negativos, por isso 65% dos pares negativos e todos os desconhecidos foram aceitos. Os thresholds não se transferem entre modelos porque cada treino muda a escala das similaridades.
  - Um threshold próximo de 0,61 separaria quase todos os pares nesse conjunto, mas escolhê-lo com estes dados seria usar um conjunto de teste (e minúsculo) para ajustar o limiar, o que o enunciado proíbe (item 29). Fica registrado como diagnóstico, não como parâmetro.
- O threshold **não foi reajustado** com dados privados, porque eles funcionam como conjunto de teste e são poucos.

## 7. Limitações
- Dataset privado muito pequeno (3 pessoas, 13 fotos, 46 pares, 7 tentativas de desconhecido): os números têm incerteza grande (por exemplo, 4 de 7 tem intervalo de confiança de 95% aproximado de 25% a 84%). São indicativos, não medidas precisas.
- O FAR de teste da LFW vem de poucos eventos: 7 falsos aceites (≈ 0,17%, IC 95% aproximado de 0,07% a 0,35%) no modelo pré-treinado.
- Pares aleatórios da LFW são fáceis e subestimam o FAR entre pessoas parecidas.
- Possível sobreposição de identidades entre VGGFace2 (pré-treino) e LFW pode inflar os resultados.
- A LFW tem pouca diversidade demográfica e de qualidade de imagem; o aviso oficial da LFW vale para o uso (ver README).
- ArcFace/MobileFaceNet como arquiteturas, o triplet loss (E04) e o ajuste completo (E05) não foram avaliados.

## 8. Custo computacional
- **Treino:** cerca de 94 s para 10 épocas na GPU T4 (10,5 milhões de parâmetros treináveis, só o último estágio). Esse treino é feito uma única vez.
- **Inferência:** a arquitetura é a mesma antes e depois do ajuste, então o custo por imagem é o mesmo; as medidas de tempo de detecção e de embedding em CPU estão em `experiments/results/e01/metrics.json` (`timing_seconds`).
- **Distribuição:** o checkpoint do E03 pesa cerca de 112 MB e não vai ao Git.

## 9. Desempenho contra complexidade
O E03 acrescenta um passo de treino e um arquivo de pesos para um ganho de cerca de 1 ponto de acurácia e 2 pontos de FRR na LFW, sem melhora nos dados privados. O modelo pré-treinado com o threshold adequado entrega quase o mesmo desempenho com menor complexidade operacional. Por isso a aplicação usa o pré-treinado (ADR-001), com o E03 disponível como opção testada.
