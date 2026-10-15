# ADR-002 — Armazenamento de embeddings

**Status**: proposto

## Contexto
Embeddings são dados biométricos privados. Precisamos de exclusão por pessoa.

## Proposta
SQLite local em `database/` (ignorado pelo Git), uma tabela de pessoas e uma de embeddings com chave estrangeira e `ON DELETE CASCADE`. Fotos em `faces/<id>/`, removidas junto com a pessoa. Sem serviços externos.

## Alternativas
Arquivos `.npy` por pessoa; banco vetorial dedicado (excessivo para o tamanho do projeto).
