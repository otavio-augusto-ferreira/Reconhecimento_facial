# 006 — Divisão dos dados e prevenção de data leakage

**Status**: rascunho

## Problema
Avaliar o reconhecimento facial exige que treino, validação e teste não compartilhem informação. Em reconhecimento facial isso inclui a própria identidade: se uma pessoa aparece em treino e teste, o modelo é avaliado em pessoas que já viu, e o resultado não representa o caso de uso (pessoas novas e desconhecidos).

## Requisitos
- R1: dividir o dataset público **por identidade** (cada pessoa pertence a exatamente um conjunto: treino, validação ou teste).
- R2: divisão determinística a partir de uma seed registrada.
- R3: usar apenas identidades com pelo menos 2 imagens (necessário para formar pares positivos).
- R4: gerar pares de verificação balanceados (mesma pessoa / pessoas diferentes) dentro de cada conjunto.
- R5: verificar automaticamente ausência de leakage: identidades em comum, caminhos repetidos e arquivos idênticos (hash) entre conjuntos.
- R6: salvar o manifesto da divisão (caminhos relativos) para reprodutibilidade.
- R7: o conjunto de teste não é usado para treino, hiperparâmetros, threshold ou escolha de modelo.

## Critérios de aceitação
- CA1: nenhuma identidade aparece em mais de um conjunto.
- CA2: mesma seed produz a mesma divisão.
- CA3: pares positivos têm a mesma identidade; negativos, identidades diferentes; proporção 50/50.
- CA4: a checagem de leakage falha quando se injeta uma identidade ou arquivo repetido.
- CA5: identidades com 1 imagem são descartadas.

## Limitações conhecidas
- A checagem por hash detecta cópias exatas, não imagens quase idênticas (recortes, recompressão). Quase-duplicatas devem ser inspecionadas por similaridade de embedding na análise de erros.
