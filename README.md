# Reconhecedor Facial

Trabalho prático da disciplina de Inteligência Artificial (UFMS). Autores: Otávio Augusto da Silva Ferreira e Vinícius Rodrigues da Silva Cardoso.

Sistema de reconhecimento facial baseado em embeddings, desenvolvido com Spec-Driven Development.

- **Parte 1**: experimentação, treinamento e fine-tuning (`experiments/`).
- **Parte 2**: aplicação web de cadastro e reconhecimento (`src/facerec/app/`).

## Privacidade e LGPD

Este projeto trata dados biométricos. **Nenhuma foto, embedding, banco biométrico, checkpoint treinado com dados privados ou termo de consentimento está neste repositório.** Veja `.gitignore` e `data/README.md`.
Ao clonar, a aplicação inicia com "Nenhuma pessoa cadastrada".

## Datasets públicos

| Dataset | Fonte oficial | Finalidade | Licença / termos | Uso no projeto |
|---|---|---|---|---|
| LFW — Labeled Faces in the Wild | https://vis-www.cs.umass.edu/lfw/ | Pesquisa em reconhecimento facial | Sem licença explícita localizada. Aviso oficial (Wayback, captura de 02/01/2024): benchmark público de pesquisa para verificação facial, sem recomendação de uso comercial. Página oficial indisponível em 20/10/2026. Arquivo obtido do espelho figshare usado pelo scikit-learn; SHA-256 conferido. | Benchmark e fine-tuning, com divisão por identidade (seed 42): 1176/252/252 identidades em treino/validação/teste. |

## Modelos pré-treinados

| Modelo | Fonte | Pesos | Licença / termos | Uso no projeto |
|---|---|---|---|---|
| FaceNet (InceptionResnetV1) | pacote `facenet-pytorch` (https://github.com/timesler/facenet-pytorch) | Treinados na VGGFace2 (o README do pacote cita também a CASIA-Webface); baixados automaticamente na primeira execução (cerca de 107 MB) e não versionados | Código do pacote: MIT (verificado na página do repositório em 25/10/2026). O README do pacote não informa a licença dos pesos nem os termos da VGGFace2, e não localizei termos oficiais da VGGFace2; usamos apenas os pesos prontos, sem baixar nem redistribuir a base. Pendente: confirmar com o professor | Embeddings de 512 dimensões: baseline e ponto de partida do fine-tuning |
| MTCNN | pacote `facenet-pytorch` | Incluídos no pacote; o README não informa a origem nem a licença dos pesos | Código do pacote: MIT. Licença dos pesos: não informada na fonte | Detecção de faces e pontos de referência |

Nota: o pré-treinamento na VGGFace2 pode ter identidades em comum com a LFW, o que pode tornar os resultados otimistas (limitação registrada).

## Resultados (Parte 1)

| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | Observação |
|---|---|---|---|---|---|---|
| E01 | FaceNet (InceptionResnetV1, VGGFace2) | pré-treinado, sem fine-tuning; threshold por acurácia na validação | público (LFW) | 0,9477 | 0,9454 | Teste, cosseno, threshold 0,4009. AUC 0,9657, EER 0,0733, FAR 0,0100, FRR 0,0947 |
| E01b | FaceNet (InceptionResnetV1, VGGFace2) | pré-treinado; threshold por FAR ≤ 0,1% na validação (adotado na aplicação) | público (LFW) | 0,9215 | 0,9150 | Teste, threshold 0,5404. FAR 0,0017, FRR 0,1553 (7 falsos aceites em 4172 pares negativos) |
| E02 | FaceNet (InceptionResnetV1, VGGFace2) | fine-tuning parcial (último estágio), perda ce, melhor época 0 | público (LFW, treino) | 0,9207 | 0,9140 | melhor época 0 = igual ao pré-treinado (o ajuste não melhorou a validação). Teste, threshold FAR ≤ 0,1% da validação (0,5404). AUC 0,9697, FAR 0,0017, FRR 0,1570 |
| E03 | FaceNet (InceptionResnetV1, VGGFace2) | fine-tuning parcial (último estágio), perda arcface, melhor época 8 | público (LFW, treino) | 0,9306 | 0,9257 | Teste, threshold FAR ≤ 0,1% da validação (0,4609). AUC 0,9714, FAR 0,0031, FRR 0,1357 |

Resultados com o dataset privado são apresentados apenas de forma agregada.

Dataset privado (apenas adultos com consentimento; resultados agregados, threshold 0.5404 fixo da LFW): 3 pessoas e 13 fotos. Verificação em 46 pares: accuracy 0,8913, F1 0,9020, AUC 0,9962, FAR 0,2174, FRR 0,0000. Identificação com cadastro de 2 fotos por pessoa: 100.0% das tentativas de pessoas cadastradas reconhecidas corretamente e 57.1% dos desconhecidos aceitos por engano.

## Análise e decisão

- **Melhor configuração da Parte 1 (pela validação): E03**, FaceNet com ajuste parcial e perda ArcFace. O ganho sobre o pré-treinado é pequeno (EER de validação 0,0507 para 0,0503; FRR de teste 0,1570 para 0,1357). O E02 (softmax de cosseno) não melhorou a validação; E04 e E05 não foram executados.
- **Aplicação (Parte 2): FaceNet pré-treinado com threshold 0,5404** (FAR <= 0,1% na validação). Desvio justificado em `docs/decisions/ADR-001-face-model.md`; o E03 está disponível em `configs/app_e03.yaml`.
- **Limitação principal:** no dataset privado (3 pessoas, 13 fotos), o threshold escolhido na LFW foi permissivo demais para desconhecidos, embora a ordenação das similaridades seja boa (AUC acima de 0,99).
- Análise completa (metodologia, impacto do threshold, erros, limitações, custo): `docs/ANALISE.md`.

## Reprodução

Veja `docs/EXECUCAO.md` (ambiente, experimentos e aplicação) e `docs/COLAB.md` (fine-tuning na GPU do Colab). Resumo:

```
pip install -r requirements.txt && pytest
python experiments/e01_facenet_baseline.py --config configs/e01_facenet_baseline.yaml
python experiments/e01b_operating_points.py --config configs/e01b_operating_points.yaml
python experiments/e02_finetune.py --config configs/e02_partial_ce.yaml      # GPU (Colab)
python -m uvicorn facerec.app.main:app --host 127.0.0.1 --port 8000          # aplicação em http://127.0.0.1:8000
```

O dataset público é obtido localmente (`data/README.md`). Nenhum dado privado faz parte do repositório.

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
