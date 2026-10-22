# Fine-tuning no Google Colab (GPU gratuita)

Use **apenas dados públicos** (LFW) na nuvem. Fotos de pessoas reais do dataset privado nunca saem do computador local.

## 1. Preparar o Colab
1. Abra https://colab.research.google.com e crie um notebook novo.
2. Menu **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)**.
3. Cada bloco abaixo é uma célula. Rode em ordem.

```python
!nvidia-smi
!python --version
```
Deve aparecer uma GPU (T4). Se a versão do Python for 3.13 ou maior, siga o bloco "Python 3.12 no Colab" no fim deste arquivo antes do passo 3.

## 2. Código
Se o repositório for **público**:
```python
!git clone https://github.com/otavio-augusto-ferreira/Reconhecimento_facial.git
%cd Reconhecimento_facial
```
Se for **privado**, no seu computador (Git Bash, na raiz do projeto) crie um zip só com os arquivos versionados e envie-o ao Colab pelo painel de arquivos:
```
git archive --format=zip --output ../codigo.zip HEAD
```
```python
!unzip -q /content/codigo.zip -d /content/Reconhecimento_facial
%cd /content/Reconhecimento_facial
```

## 3. Dependências
```python
!pip install -q -r requirements.txt
!pip install -q -e .
!pytest -q tests/test_training.py tests/test_crops_evaluation.py tests/test_metrics.py
```
O `pytest` deve passar. (O `requirements.txt` fixa as versões; o `facenet-pytorch` exige `torch` 2.2.x.)

## 4. LFW (dado público)
```python
!mkdir -p data/public
!curl -sL -o lfw.tgz https://ndownloader.figshare.com/files/5976018
!sha256sum lfw.tgz
!tar -xzf lfw.tgz -C data/public && rm lfw.tgz
!ls data/public/lfw | wc -l
```
O SHA-256 deve ser `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0` e a última linha deve mostrar `5749` pastas. O manifesto da divisão (`experiments/splits/lfw_seed42.json`) já vem no repositório: **não gere outro**, senão os pares mudam e a comparação com E01/E01b deixa de valer.

## 5. Teste rápido (2 minutos)
```python
!python experiments/e02_finetune.py --config configs/e02_partial_ce.yaml --limit-identities 20 --epochs 2
```
Deve imprimir `dispositivo: cuda`, a época 0 (pré-treinado), as épocas de treino e as duas tabelas (baseline e fine-tuning).

## 6. Experimentos
Rode um por vez e guarde a saída de cada um. Prioridade: E02, depois E03.

```python
!python experiments/e02_finetune.py --config configs/e02_partial_ce.yaml
!python experiments/e02_finetune.py --config configs/e03_partial_arcface.yaml
!python experiments/e02_finetune.py --config configs/e04_partial_triplet.yaml
!python experiments/e02_finetune.py --config configs/e05_full_arcface.yaml
```
Na primeira execução ele detecta e recorta as faces e guarda o cache em `data/public/cache/`; as seguintes reaproveitam. O treino leva alguns minutos por época na T4.

O que olhar na saída:
- **melhor época = 0** significa que nenhuma época melhorou a validação em relação ao modelo pré-treinado. É um resultado válido: registre-o e não force o fine-tuning.
- A tabela de pontos de operação mostra o **threshold escolhido na validação** e o FAR/FRR no teste. Compare a linha `far_0.1pct` do baseline com a do fine-tuning: é a que a aplicação usa (ADR-003).

## 7. Trazer os resultados (sem os pesos)
```python
!zip -r resultados.zip experiments/results -x "*.pt"
from google.colab import files
files.download("resultados.zip")
```
Extraia em `experiments/results/` no repositório local. Os pesos (`checkpoints/public/*.pt`) **não** vão ao Git.

## Regras
- A sessão do Colab é apagada ao fechar: baixe os resultados antes de sair. Se quiser manter o cache de recortes, monte o Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) e use `--cache-dir /content/drive/MyDrive/facerec_cache`.
- Nunca coloque o token da LLM no notebook.
- Cuidado com limites de uso da GPU gratuita: se a sessão cair, rode de novo; o cache poupa a parte lenta.

## Python 3.12 no Colab (só se o Python do Colab for 3.13 ou maior)
O `facenet-pytorch` 2.6.0 exige `torch` 2.2.x, que não tem versão para Python 3.13. Crie um ambiente com Python 3.12:
```python
!pip install -q uv
!uv venv --python 3.12 /content/venv312
!uv pip install --python /content/venv312/bin/python -r requirements.txt -e .
```
Depois use `!/content/venv312/bin/python` no lugar de `!python` (e `!/content/venv312/bin/python -m pytest`) em todos os comandos acima.
