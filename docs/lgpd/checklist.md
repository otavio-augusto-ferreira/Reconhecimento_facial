# Checklist de conformidade (LGPD) — marcar com [x] e commitar só este arquivo, sem dados

## Antes de coletar
- [ ] Todos os participantes são adultos e assinaram o Anexo A (docs/lgpd/termo-consentimento.md), cada um o seu, com a assinatura do aluno responsável; as vias assinadas ficam em `data/private/_consent/` ou `consent/`, fora do Git
- [ ] A opção de vídeo (autoriza / não autoriza) está marcada em todos os termos
- [ ] Cada pessoa tem um id anônimo (p01, p02...); a tabela id↔nome só existe no papel ou em `consent/`
- [ ] Nenhuma foto vem de redes sociais, de terceiros ou de pessoas que não consentiram
- [ ] Combinado com cada participante que os dados serão excluídos ao fim da avaliação (o Anexo A não traz prazo; registrar a data combinada)

## Durante
- [ ] Fotos só em `data/private/` (ignorado pelo Git); banco em `database/`; logs em `logs/`
- [ ] Nenhuma foto, embedding, banco, checkpoint ou termo foi enviado a serviço externo, nuvem (Colab, Drive) ou agente de IA, incluindo este assistente e o Qwen do professor
- [ ] Fine-tuning no Colab usa somente o LFW; dados privados nunca vão ao Colab
- [ ] O limiar não foi ajustado com dados privados (vem da validação da LFW)
- [ ] Logs da aplicação não têm nomes nem biometria (verificado por teste)

## Antes de cada commit
- [ ] `git status` não lista `data/`, `database/`, `logs/`, `consent/`, `*.db`, `*.pt`, fotos nem `.env`
- [ ] `git diff --cached --stat` conferido arquivo por arquivo
- [ ] Nos vídeos e capturas de tela só aparecem pessoas que marcaram AUTORIZO na demonstração em vídeo; as demais não aparecem (nem em fotos, nem na câmera ao fundo)
- [ ] Nenhum modelo treinado com dados privados foi publicado (item 5 do termo)

## Ao final
- [ ] Apagados: `data/private/`, `database/`, `logs/`, `checkpoints/`, `experiments/results/private/` (se houver dados detalhados) e caches
- [ ] Termos assinados guardados ou destruídos conforme o prazo informado
- [ ] Conferido no GitHub (histórico incluso) que nenhum dado privado foi enviado
- [ ] README descreve apenas resultados agregados do dataset privado
