# Dados

Este diretório **não contém dados**. Cada usuário obtém os dados localmente.

## Dataset público (LFW)

Baixe diretamente da fonte oficial (https://vis-www.cs.umass.edu/lfw/) e extraia em `data/public/lfw/`.
Confira os termos de uso antes. Não redistribuímos cópias.

## Dataset privado (fotos dos participantes)

Criado localmente por cada grupo, apenas com pessoas que assinaram o Termo de Consentimento.

- Fotos em `data/private/<id_anonimo>/` (ignorado pelo Git).
- Termos de consentimento guardados fora do repositório.
- Processamento somente local: não enviar fotos nem embeddings a serviços externos (nuvem, agentes de IA, APIs).
- Exclusão de uma pessoa: ver `docs/specs/002-person-enrollment.md`.
- Ao fim da disciplina, apagar fotos, embeddings, bancos e checkpoints privados.
