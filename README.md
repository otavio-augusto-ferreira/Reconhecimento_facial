# Reconhecedor Facial

Trabalho prático da disciplina de Inteligência Artificial (UFMS). Autores: Tavinho e Vinicius.

Sistema de reconhecimento facial baseado em embeddings, desenvolvido com Spec-Driven Development.

- **Parte 1**: experimentação, treinamento e fine-tuning (`experiments/`).
- **Parte 2**: aplicação web de cadastro e reconhecimento (`src/app/`).

## Privacidade e LGPD

Este projeto trata dados biométricos. **Nenhuma foto, embedding, banco biométrico, checkpoint treinado com dados privados ou termo de consentimento está neste repositório.** Veja `.gitignore` e `data/README.md`.
Ao clonar, a aplicação inicia com "Nenhuma pessoa cadastrada".

## Datasets públicos

| Dataset | Fonte oficial | Finalidade | Licença / termos | Uso no projeto |
|---|---|---|---|---|
| LFW — Labeled Faces in the Wild | https://vis-www.cs.umass.edu/lfw/ | Pesquisa em reconhecimento facial | Sem licença explícita localizada. Aviso oficial (Wayback, captura de 02/01/2024): benchmark público de pesquisa para verificação facial, sem recomendação de uso comercial. Página oficial indisponível em 20/10/2026. Arquivo obtido do espelho figshare usado pelo scikit-learn; SHA-256 conferido. | Benchmark e fine-tuning, com divisão por identidade (seed 42): 1176/252/252 identidades em treino/validação/teste. |

## Modelos pré-treinados

TODO: listar modelo, fonte e licença.

## Resultados (Parte 1)

| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | Observação |
|---|---|---|---|---|---|---|
| E01 | FaceNet (InceptionResnetV1, VGGFace2) | pré-treinado, sem fine-tuning | público (LFW) | 0,9477 | 0,9454 | Teste, cosseno, threshold 0,4009 escolhido na validação. AUC 0,9657, EER 0,0733, FAR 0,0100, FRR 0,0947 |

Resultados com o dataset privado são apresentados apenas de forma agregada.

## Reprodução

TODO: instruções de ambiente, download dos datasets e execução dos experimentos.

## Estrutura

```
src/            código-fonte
experiments/    scripts e configs dos experimentos
configs/        configurações externas
tests/          testes
docs/specs/     especificações (Spec-Driven Development)
docs/decisions/ ADRs
data/README.md  instruções de dados
```
