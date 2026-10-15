# 005 — Interface web com câmera

**Status**: rascunho

## Problema
Interface web para cadastrar, reconhecer e excluir pessoas usando a câmera do navegador.

## Requisitos
- R1: página com webcam e feedback visual (caixa + nome ou "desconhecido").
- R2: telas/ações de cadastro, reconhecimento e exclusão.
- R3: backend (FastAPI) separado da interface, com módulos de detecção, embedding e armazenamento.
- R4: erros exibidos de forma clara (sem face, múltiplas faces, câmera indisponível).
- R5: logs sem dados biométricos.
- R6: execução totalmente local.

## Critérios de aceitação
- CA1: fluxo cadastrar → reconhecer → excluir funciona de ponta a ponta.
- CA2: erros de R4 reproduzíveis na demonstração.
- CA3: instruções de execução no README.
