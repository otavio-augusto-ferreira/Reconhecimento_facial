# 005 — Interface web com câmera

**Status**: implementada

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

## Implementação
- Backend: `src/facerec/app/` (`api.py`, `service.py`, `storage.py`, `config.py`, `errors.py`), separado da interface.
- Interface: `src/facerec/app/static/` (HTML, CSS e JavaScript puros, sem recursos externos). Três seções: Reconhecer (câmera contínua, caixa e nome ou "Desconhecido"), Cadastrar (fotos da câmera ou arquivos) e Pessoas cadastradas (lista e exclusão com confirmação).
- Erros mostrados na tela: sem face, múltiplas faces, baixa qualidade, imagem inválida ou grande demais, nome vazio ou repetido, cadastro duplicado (pede confirmação), câmera negada, não encontrada ou em uso, e servidor indisponível.
- Logs em `logs/app.log`: eventos e contagens, sem nomes nem dados biométricos.
- Como executar e roteiro de teste manual: `docs/EXECUCAO.md`.
Testes: `tests/test_app_api.py`.
