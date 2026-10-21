# Como executar

Testado com Python 3.12 (64 bits). Os comandos abaixo são para o Git Bash, na raiz do projeto.

## 1. Ambiente

```
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash). No Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                               # todos os testes devem passar
```

## 2. Parte 1 — experimentos (dataset público)

O LFW fica em `data/public/lfw` (não vai ao Git). Veja `data/README.md`.

```
python experiments/make_lfw_splits.py --root data/public/lfw    # divisão por identidade (já versionada em experiments/splits/)
python experiments/e01_facenet_baseline.py --config configs/e01_facenet_baseline.yaml
python experiments/e01b_operating_points.py --config configs/e01b_operating_points.yaml
```

Resultados agregados em `experiments/results/`. Para um teste rápido do E01, acrescente `--limit-identities 20`.

## 3. Parte 2 — aplicação web (tudo local)

```
python -m uvicorn facerec.app.main:app --host 127.0.0.1 --port 8000
```

Abra **http://127.0.0.1:8000** no navegador (a câmera só funciona em `localhost`/`127.0.0.1` ou HTTPS). Na primeira execução os pesos do modelo são baixados (cerca de 107 MB). O servidor escuta apenas neste computador.

Configuração: `configs/app.yaml` (threshold, qualidade mínima do cadastro, limites de imagem, caminhos). Outro arquivo pode ser usado com a variável `FACEREC_CONFIG`.

Arquivos criados em execução, todos ignorados pelo Git e todos dados sensíveis:
- `database/facerec.db`: pessoas e embeddings (nenhuma foto é guardada);
- `logs/app.log`: eventos e contagens, sem nomes nem dados biométricos.

**Ao terminar a avaliação ou a gravação dos vídeos, exclua as pessoas pela interface e apague a pasta `database/`.**

## 4. Roteiro de teste manual (use para a demonstração em vídeo)

1. Repositório recém-clonado: a aba "Pessoas cadastradas" mostra "Nenhuma pessoa cadastrada".
2. Cadastrar: ligar a câmera, aba "Cadastrar", nome, 3 a 5 fotos com poses diferentes, "Cadastrar".
3. Reconhecer: aba "Reconhecer", "Iniciar reconhecimento": a caixa verde mostra o nome e a similaridade.
4. Desconhecido: uma pessoa não cadastrada na frente da câmera aparece como "Desconhecido".
5. Erros: foto sem rosto (câmera tapada), foto com duas pessoas no cadastro, foto distante ou escura, nome vazio, nome repetido, cadastro duplicado (mesma pessoa com outro nome), câmera negada nas permissões do navegador, servidor parado.
6. Excluir: aba "Pessoas cadastradas" → "Excluir" → confirmar; depois a pessoa aparece como "Desconhecido".

## 5. Privacidade (LGPD)

- Use apenas pessoas adultas que deram consentimento por escrito; os termos ficam fora do repositório (`consent/`).
- Não envie fotos, embeddings ou o banco a serviços externos, incluindo assistentes de IA.
- Antes de cada commit: `git status` não pode listar `data/`, `database/`, `logs/`, `faces/`, `consent/`, `*.db`, `*.pt` ou `*.pth`.
