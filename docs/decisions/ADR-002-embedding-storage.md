# ADR-002 — Armazenamento de embeddings

**Status**: aceito

## Contexto
Embeddings são dados biométricos privados. Precisamos de exclusão completa por pessoa e de nenhum dado fora do computador local.

## Decisão
- SQLite local em `database/facerec.db` (ignorado pelo Git): tabela `persons` e tabela `embeddings` com chave estrangeira e `ON DELETE CASCADE`.
- **Fotos não são armazenadas.** A imagem enviada existe só na memória durante a requisição; o banco guarda apenas os vetores. Isso muda a proposta inicial (pasta `faces/<id>/`): menos dados sensíveis, e a exclusão fica mais simples.
- Exclusão: `DELETE` em cascata, `PRAGMA secure_delete = ON` (sobrescreve o conteúdo apagado) e `VACUUM` (devolve o espaço ao sistema de arquivos). Um teste confirma que os bytes do vetor não permanecem no arquivo do banco depois de excluir a pessoa.
- Nome único, sem diferenciar maiúsculas de minúsculas.
- Logs (`logs/app.log`, ignorado pelo Git) registram eventos e contagens, nunca nomes, imagens ou embeddings. Um teste verifica isso.
- Sem serviços externos.

## Alternativas
Arquivos `.npy` por pessoa (exclusão por arquivo, mas sem integridade referencial); banco vetorial dedicado (excessivo para o tamanho do projeto).

## Consequências
- A busca compara com todos os embeddings em memória (produto matricial). Para dezenas de pessoas isso é imediato; para milhares, seria preciso um índice.
- Sem fotos guardadas, não é possível recalcular os embeddings se o modelo mudar: nesse caso, as pessoas precisam ser cadastradas de novo.
- O arquivo do banco é um dado biométrico: não deve ser versionado, copiado ou enviado a serviços externos.
