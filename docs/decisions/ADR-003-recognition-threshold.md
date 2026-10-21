# ADR-003 — Threshold de reconhecimento

**Status**: aceito

## Contexto
A aplicação precisa decidir entre "reconhecido" e "desconhecido" (spec 004). A decisão depende de um threshold sobre a similaridade de cosseno entre o rosto e as pessoas cadastradas.

O E01 escolheu 0,4009 maximizando a acurácia na validação, em pares aleatórios da LFW. Ao testar a aplicação com modelos reais, apareceram falsos aceites entre pessoas com aparência semelhante (similaridade de 0,44 a 0,58 entre pessoas diferentes), o que mostrou que esse critério é permissivo demais para uso com várias pessoas cadastradas. Os detalhes e o raciocínio estão em `docs/experiments/E01b-operating-points.md`.

## Método
- O threshold é escolhido **somente no conjunto de validação**; o teste só reporta FAR/FRR.
- Critério candidato para a aplicação: **menor FRR com FAR ≤ 0,1% na validação** (`select_threshold(..., "far", far_target=0.001)`). Aceitar um desconhecido é o erro mais grave, e o risco cresce com o número de pessoas cadastradas (`1 - (1 - FAR)^N`).
- O E01b compara quatro critérios (acurácia máxima, EER, FAR ≤ 1%, FAR ≤ 0,1%) e mostra o custo de cada um em falsos rejeites.

## Alternativas
| Critério | Vantagem | Desvantagem |
| --- | --- | --- |
| Acurácia máxima (E01) | Melhor acerto geral em pares balanceados | Ignora que os erros têm custos diferentes |
| EER | Erros balanceados | Também trata os dois erros como iguais |
| FAR ≤ 1% | Menos falsos aceites | Com N = 10 pessoas cadastradas, ~10% dos desconhecidos ainda passariam |
| FAR ≤ 0,1% | Poucos falsos aceites mesmo com N grande | FRR maior: a pessoa cadastrada precisa tentar mais de uma vez |

## Decisão
A aplicação usa o critério **menor FRR com FAR ≤ 0,1% na validação**: threshold **0,5404** (`configs/app.yaml`, `recognition.threshold`), obtido no E01b (LFW, cosseno, seed 42).

| Critério (E01b) | Threshold | Teste FAR | Teste FRR | Desconhecido aceito, N = 10 |
| --- | --- | --- | --- | --- |
| Acurácia máxima | 0,4283 | 0,0077 | 0,0954 | 7,4% |
| EER | 0,2986 | 0,0460 | 0,0712 | 37,6% |
| FAR ≤ 1% | 0,4144 | 0,0093 | 0,0920 | 9,0% |
| **FAR ≤ 0,1%** | **0,5404** | **0,0017** | **0,1553** | **1,7%** |

Motivo: aceitar um desconhecido é o erro mais grave e o risco cresce com o número de pessoas cadastradas. O custo é um FRR maior (0,155 contra 0,095 em pares isolados). O threshold anterior (0,4009, do E01) deixou de ser usado.

## Consequências
- O valor vive em configuração externa; trocá-lo não exige mudar código.
- Um threshold mais rígido aumenta os falsos rejeites: o cadastro com 3 a 5 fotos variadas (spec 002) e o reconhecimento contínuo (várias tentativas por segundo) reduzem esse efeito.
- O reconhecimento contínuo também dá várias chances de falso aceite a um desconhecido que permanece diante da câmera. Uma melhoria possível é exigir a mesma identidade em alguns quadros seguidos antes de exibir o nome.
- Limitação a declarar: o FAR medido em pares aleatórios subestima o FAR entre pessoas parecidas; na verificação informal, o par mais parecido (0,576) ainda passaria de 0,5404. O FAR de 0,17% no teste vem de apenas 7 falsos aceites (intervalo de 95% aproximado de 0,07% a 0,35%). Uma melhoria possível, fora do escopo, é exigir também uma margem entre a melhor e a segunda melhor pessoa.
