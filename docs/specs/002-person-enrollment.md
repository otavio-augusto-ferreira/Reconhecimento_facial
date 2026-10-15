# 002 — Cadastro e exclusão de pessoa

**Status**: rascunho

## Problema
Cadastrar uma pessoa a partir de imagens e permitir remover todos os seus dados (Privacy by Design).

## Requisitos
- R1: cadastro com nome/identificador e uma ou mais imagens.
- R2: pipeline: imagem → detecção → alinhamento → embedding → armazenamento local.
- R3: rejeitar imagem sem face, com múltiplas faces ou de baixa qualidade.
- R4: detectar cadastro duplicado (embedding muito similar a pessoa já cadastrada).
- R5: exclusão de pessoa remove cadastro, embeddings, fotos armazenadas e dados associados.
- R6: banco local fora do Git; inicialmente vazio.

## Critérios de aceitação
- CA1: cadastro válido cria a pessoa e seus embeddings.
- CA2: cada caso de rejeição retorna mensagem de erro específica.
- CA3: cadastro duplicado é sinalizado e exige confirmação.
- CA4: após excluir, a pessoa não é mais reconhecida e nenhum arquivo dela permanece.
- CA5: repositório recém-clonado mostra "Nenhuma pessoa cadastrada".
