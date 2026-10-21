# 002 — Cadastro e exclusão de pessoa

**Status**: implementada

## Problema
Cadastrar uma pessoa a partir de imagens e permitir remover todos os seus dados (Privacy by Design).

## Requisitos
- R1: cadastro com nome/identificador e uma ou mais imagens.
- R2: pipeline: imagem → detecção → alinhamento → embedding → armazenamento local.
- R3: rejeitar imagem sem face, com múltiplas faces ou de baixa qualidade (confiança < 0,97 ou face com menos de 80 px; valores em `configs/app.yaml`).
- R4: detectar cadastro duplicado (embedding muito similar a pessoa já cadastrada).
- R5: exclusão de pessoa remove cadastro, embeddings e dados associados. Fotos não são armazenadas (ADR-002).
- R6: banco local fora do Git; inicialmente vazio.

## Critérios de aceitação
- CA1: cadastro válido cria a pessoa e seus embeddings.
- CA2: cada caso de rejeição retorna mensagem de erro específica.
- CA3: cadastro duplicado é sinalizado e exige confirmação.
- CA4: após excluir, a pessoa não é mais reconhecida e nenhum arquivo dela permanece.
- CA5: repositório recém-clonado mostra "Nenhuma pessoa cadastrada".

## Implementação
`src/facerec/app/service.py` (`FaceService.enroll` e `delete_person`) e `src/facerec/app/storage.py`. Também rejeita nome vazio ou repetido, imagens ilegíveis ou grandes demais, mais de 10 imagens, e imagens do mesmo cadastro que parecem ser de pessoas diferentes (cada imagem é comparada com a média das outras).
Testes: `tests/test_app_service.py`, `tests/test_app_storage.py` (inclui a verificação dos bytes no arquivo do banco após a exclusão).
