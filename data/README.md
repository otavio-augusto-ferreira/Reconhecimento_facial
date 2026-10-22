# Dados

Este diretório **não contém dados**. Cada usuário obtém os dados localmente.

## Dataset público (LFW)

Fonte oficial: https://vis-www.cs.umass.edu/lfw/ (fora do ar em 20/10/2026; veja o aviso do Wayback Machine no README).
Confira os termos de uso quando a página voltar. Não redistribuímos cópias.

Enquanto isso usamos o espelho do figshare (o mesmo que o scikit-learn usa) e conferimos o SHA-256 do arquivo:

```
mkdir -p data/public
curl -L -o lfw.tgz https://ndownloader.figshare.com/files/5976018
sha256sum lfw.tgz        # deve ser 055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0
tar -xzf lfw.tgz -C data/public
```

Isso cria `data/public/lfw/<nome_da_pessoa>/*.jpg`. A pasta `data/public/` é ignorada pelo Git.

## Dataset privado (fotos dos participantes)

Criado localmente por cada grupo, apenas com pessoas que assinaram o Termo de Consentimento.

- Fotos em `data/private/<id_anonimo>/` (ignorado pelo Git).
- Termos de consentimento guardados fora do repositório.
- Processamento somente local: não enviar fotos nem embeddings a serviços externos (nuvem, agentes de IA, APIs).
- Exclusão de uma pessoa: ver `docs/specs/002-person-enrollment.md`.
- Ao fim da disciplina, apagar fotos, embeddings, bancos e checkpoints privados.
