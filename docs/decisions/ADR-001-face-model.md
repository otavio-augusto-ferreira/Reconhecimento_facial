# ADR-001 — Modelo de embedding facial

**Status**: aceito (com desvio justificado entre o melhor da Parte 1 e o usado na aplicação)

## Contexto
Sem GPU local; fine-tuning só com dados públicos (LFW) em GPU gratuita de nuvem (T4). A aplicação roda em CPU. Dados privados nunca saem do computador local (ver `docs/lgpd/checklist.md`).

## Candidatos
- FaceNet (InceptionResnetV1, pesos VGGFace2, pacote `facenet-pytorch`): **avaliado** (E01, E01b, E02, E03).
- ArcFace / MobileFaceNet (insightface): **não avaliados**. Limitação: o pacote usado já traz detector e modelo de forma integrada, o tempo era curto e o fine-tuning com perda ArcFace foi investigado sobre o FaceNet (E03). A comparação entre arquiteturas fica como trabalho futuro.

## Critérios
Desempenho na **validação** (EER, AUC, FRR com FAR ≤ 0,1%), custo em CPU, reprodutibilidade (o repositório não pode conter pesos derivados que não possa redistribuir) e comportamento no domínio da aplicação.

## Evidências (LFW; mesmos pares; seed 42)
| Modelo | Época | Val AUC | Val EER | Val FRR (FAR ≤ 0,1%) | Teste AUC | Teste FAR | Teste FRR | Teste acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pré-treinado (= E02, época 0) | 0 | 0,9828 | 0,0507 | 0,0995 | 0,9697 | 0,0017 | 0,1570 | 0,9207 |
| E03 — ArcFace, ajuste parcial | 8 | 0,9858 | 0,0503 | 0,0918 | 0,9714 | 0,0031 | 0,1357 | 0,9306 |

E02 (softmax de cosseno) não melhorou a validação: a melhor época foi a 0. Detalhes em `docs/experiments/E02-finetune.md` e `docs/ANALISE.md`.

## Decisão
1. **Melhor configuração da Parte 1 (pelo protocolo, escolhida na validação): E03.** O ganho é pequeno: o EER de validação passou de 0,0507 para 0,0503; o benefício aparece no AUC e no FRR com FAR ≤ 0,1%.
2. **Modelo da aplicação (Parte 2): FaceNet pré-treinado, threshold 0,5404 (ADR-003).** Isto é um desvio do item anterior, permitido pelo enunciado (item 36) desde que justificado:
   - o ganho do E03 na validação é marginal (ΔEER = 0,0004) e o FAR no teste foi maior (0,0031 contra 0,0017);
   - o E03 exige um checkpoint de cerca de 112 MB que não pode ir ao Git; quem clonar o projeto teria de treiná-lo antes de usar a aplicação;
   - numa verificação no dataset privado (3 pessoas, 13 fotos, resultados agregados; evidência fraca, usada apenas para a decisão de implantação e não para escolher o modelo da Parte 1), o E03 com o threshold dele (0,4609) aceitou 7 de 7 desconhecidos simulados, contra 4 de 7 do modelo pré-treinado com 0,5404. Os dois resultados são ruins, e o do E03 é pior.
3. O suporte ao E03 existe e está testado: `configs/app_e03.yaml` (checkpoint em `checkpoints/public/e03.pt` e threshold 0,4609). Se o checkpoint não existir, a aplicação falha com mensagem clara em vez de usar o threshold errado.

## Consequências
- A aplicação funciona logo após clonar, sem baixar nem treinar nada além dos pesos públicos do pacote.
- O threshold depende do modelo: ao trocar de modelo, é obrigatório trocar o threshold (cada um foi escolhido na validação do próprio modelo).
- Limitação declarada: o threshold escolhido na LFW foi permissivo demais no dataset privado (ver `docs/ANALISE.md`). Não foi reajustado com dados privados, para não usar um conjunto de teste na escolha do threshold.
