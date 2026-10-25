# Checklist de conformidade (LGPD) — marcar com [x] e commitar só este arquivo, sem dados

## Antes de coletar
- [X] Todos os participantes são adultos e assinaram o Anexo A (docs/lgpd/termo-consentimento.md), cada um o seu, com a assinatura do aluno responsável; as vias assinadas ficam em `data/private/_consent/` ou `consent/`, fora do Git
- [X] A opção de vídeo (autoriza / não autoriza) está marcada em todos os termos
- [X] Cada pessoa tem um id anônimo (p01, p02...); a tabela id↔nome só existe no papel ou em `consent/`
- [X] Nenhuma foto vem de redes sociais, de terceiros ou de pessoas que não consentiram
- [X] Combinado com cada participante que os dados serão excluídos ao fim da avaliação (o Anexo A não traz prazo; registrar a data combinada)

## Durante
- [X] Fotos só em `data/private/` (ignorado pelo Git); banco em `database/`; logs em `logs/`
- [X] Nenhuma foto, embedding, banco, checkpoint ou termo foi enviado a serviço externo, nuvem (Colab, Drive) ou agente de IA, incluindo este assistente e o Qwen do professor
- [X] Fine-tuning no Colab usa somente o LFW; dados privados nunca vão ao Colab
- [X] O limiar não foi ajustado com dados privados (vem da validação da LFW)
- [X] Logs da aplicação não têm nomes nem biometria (verificado por teste)

## Antes de cada commit
- [X] `git status` não lista `data/`, `database/`, `logs/`, `consent/`, `*.db`, `*.pt`, fotos nem `.env`
- [X] `git diff --cached --stat` conferido arquivo por arquivo
- [X] Nos vídeos e capturas de tela só aparecem pessoas que marcaram AUTORIZO na demonstração em vídeo; as demais não aparecem (nem em fotos, nem na câmera ao fundo)
- [X] Nenhum modelo treinado com dados privados foi publicado (item 5 do termo)

## Ao final
- [X] Apagados: `data/private/`, `database/`, `logs/`, `checkpoints/`, `experiments/results/private/` (se houver dados detalhados) e caches
- [X] Termos assinados guardados ou destruídos conforme o prazo informado
- [X] Conferido no GitHub (histórico incluso) que nenhum dado privado foi enviado
- [X] README descreve apenas resultados agregados do dataset privado
