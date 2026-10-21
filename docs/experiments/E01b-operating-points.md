# E01b — Pontos de operação do FaceNet pré-treinado

## Pergunta
O threshold do E01 (acurácia máxima nos pares da LFW) é adequado para a aplicação? Quanto custa, em falsos rejeites, exigir menos falsos aceites?

## Motivação
No E01 o threshold foi escolhido para maximizar a acurácia em pares aleatórios (50% mesma pessoa, 50% pessoas diferentes). Dois fatos tornam esse critério insuficiente para a aplicação:

1. **Os erros têm custos diferentes.** Aceitar um desconhecido como se fosse uma pessoa cadastrada é mais grave que pedir uma nova tentativa a quem está cadastrado.
2. **O risco cresce com o número de pessoas cadastradas.** O reconhecimento compara o rosto com N pessoas; se o FAR por par é `f`, a chance de um desconhecido ser aceito por alguma delas é, aproximadamente, `1 - (1 - f)^N`. Com `f = 1%` e N = 10, isso dá cerca de 10%.

Observação informal que reforça o ponto (**não faz parte dos resultados**): com 5 fotos públicas de celebridades, 4 das 10 combinações de pessoas diferentes passaram de 0,40 de similaridade, e todas eram pessoas com aparência semelhante (mesmo sexo, idade e estilo). Pares aleatórios da LFW são, em sua maioria, fáceis; o FAR medido neles tende a subestimar o FAR entre pessoas parecidas.

## Hipóteses
- **H1**: exigir FAR ≤ 0,1% na validação reduz os falsos aceites do teste em pelo menos 5 vezes em relação ao E01, ao custo de um FRR maior.
- **H2**: o FAR obtido no teste para cada ponto de operação fica dentro do esperado pela validação (diferença absoluta ≤ 0,005 para o alvo de 1%).

## Método
Mesmos modelo, detector, divisões (`lfw_seed42.json`) e cache de embeddings do E01. Mudanças:

- mais pares (até 40 000 por conjunto, balanceados), para resolver FAR pequenos;
- quatro critérios para escolher o threshold, **sempre só na validação**: `accuracy`, `eer`, `far ≤ 1%` e `far ≤ 0,1%`;
- cada threshold é aplicado, sem ajuste, ao teste;
- estimativa da chance de um desconhecido ser aceito por alguma entre N = 5, 10 e 20 pessoas cadastradas.

## Como reproduzir
Depois de rodar o E01 (o cache de embeddings em `data/public/cache` é reaproveitado):

```
python experiments/e01b_operating_points.py --config configs/e01b_operating_points.yaml
```

Saída: tabela impressa e `experiments/results/e01b/metrics.json`.

## Cuidados metodológicos
- O teste não escolhe nada: todos os thresholds vêm da validação.
- Com FAR de 0,1% e ~20 000 negativos, o teste tem cerca de 20 falsos aceites esperados; a incerteza é grande. Reporte os números absolutos (`confusion`), não só as taxas.
- Pares sorteados de identidades diferentes continuam sendo "fáceis". Este experimento não mede o FAR entre pessoas parecidas; isso é uma limitação a declarar no relatório.
- A LFW tem pouca diversidade (ver aviso oficial no README); os resultados não valem para outros grupos.

## Resultado
Execução de 21/10/2026, LFW inteiro, seed 42, CPU, cosseno. Pares: validação 17 562 (8 781 negativos), teste 8 344 (4 172 negativos). Todos os thresholds foram escolhidos na validação.

| Ponto de operação | Threshold | Val FAR | Val FRR | Teste FAR | Teste FRR | Falsos aceites (teste) |
| --- | --- | --- | --- | --- | --- | --- |
| Acurácia máxima | 0,4283 | 0,0071 | 0,0688 | 0,0077 | 0,0954 | 32 de 4172 |
| EER | 0,2986 | 0,0507 | 0,0507 | 0,0460 | 0,0712 | 192 de 4172 |
| FAR ≤ 1% | 0,4144 | 0,0099 | 0,0670 | 0,0093 | 0,0920 | 39 de 4172 |
| FAR ≤ 0,1% | 0,5404 | 0,0009 | 0,0995 | 0,0017 | 0,1553 | 7 de 4172 |

Chance de um desconhecido ser aceito por alguma entre N pessoas cadastradas (FAR do teste, pares independentes):

| Ponto de operação | N = 5 | N = 10 | N = 20 |
| --- | --- | --- | --- |
| Acurácia máxima | 0,038 | 0,074 | 0,143 |
| EER | 0,210 | 0,376 | 0,610 |
| FAR ≤ 1% | 0,046 | 0,090 | 0,171 |
| FAR ≤ 0,1% | 0,008 | 0,017 | 0,033 |

## Interpretação
- **H1: confirmada com ressalva.** Em relação ao E01 (FAR de teste 0,0100), o ponto FAR ≤ 0,1% reduz os falsos aceites 5,9 vezes (0,0017). Mas, comparando nos mesmos 8 344 pares, a redução frente à acurácia máxima é de 4,5 vezes (0,0077 → 0,0017), um pouco abaixo de 5. Além disso, 7 falsos aceites é um número pequeno: o intervalo de 95% (Poisson exato) vai de cerca de 0,07% a 0,35%.
- **H2: confirmada.** O FAR do teste ficou a 0,0006 do da validação no alvo de 1% (0,0093 contra 0,0099) e a 0,0008 no alvo de 0,1% (0,0017 contra 0,0009).
- **O critério EER não serve para a aplicação.** Equilibra os dois erros, mas aceita 4,6% dos pares de pessoas diferentes; com 10 pessoas cadastradas, mais de um terço dos desconhecidos seria aceito.
- **Custo do FAR ≤ 0,1%:** o FRR do teste sobe de 0,095 para 0,155 (cerca de 1 em 6,4 pares de uma mesma pessoa é rejeitado). Na aplicação, o reconhecimento é contínuo e a pessoa cadastrada tem várias chances por segundo; ainda assim, esse é o preço de aceitar menos desconhecidos.
- **O threshold depende da amostra de pares.** O ponto de acurácia máxima foi 0,4009 no E01 (3 000 pares) e 0,4283 aqui (17 562 pares), com as mesmas imagens e divisões. Não se deve tratar o threshold como valor exato; o que importa é o FAR-alvo.
- **Limitação:** os pares negativos são sorteados entre identidades quaisquer, em sua maioria fáceis. O FAR real entre pessoas parecidas é maior. Na verificação informal que motivou este experimento, o par mais parecido (0,576) ainda passaria de 0,5404.
- **Decisão:** adotar o ponto FAR ≤ 0,1% na aplicação (ADR-003).
