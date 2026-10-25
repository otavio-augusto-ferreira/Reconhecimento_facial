"""A aplicação carrega o checkpoint fine-tunado (mesmo formato salvo por experiments/e02_finetune.py)."""
import numpy as np
import pytest
import torch
from PIL import Image

from facerec.app.config import load_config
from facerec.models.facenet import FaceNetEmbedder


@pytest.fixture(scope="module")
def faces():
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)) for _ in range(2)]


def test_checkpoint_altera_os_embeddings_e_e_reprodutivel(tmp_path, faces):
    base = FaceNetEmbedder(pretrained="vggface2")
    state = {k: v.clone() for k, v in base.net.state_dict().items()}
    state["last_linear.weight"] = state["last_linear.weight"] * 0.5 + 0.01  # simula pesos ajustados
    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": state}, path)

    tuned = FaceNetEmbedder(pretrained="vggface2", checkpoint=str(path))
    a, b = base.embed(faces), tuned.embed(faces)
    assert a.shape == b.shape == (2, 512)
    assert not np.allclose(a, b, atol=1e-4)
    assert np.allclose(b, FaceNetEmbedder(pretrained="vggface2", checkpoint=str(path)).embed(faces), atol=1e-6)
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-5)


def test_checkpoint_incompativel_falha_em_vez_de_ignorar(tmp_path):
    path = tmp_path / "ruim.pt"
    torch.save({"state_dict": {"camada_inexistente": torch.zeros(1)}}, path)
    with pytest.raises(RuntimeError):
        FaceNetEmbedder(pretrained="vggface2", checkpoint=str(path))


def test_config_le_o_checkpoint(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  pretrained: vggface2\n  checkpoint: checkpoints/public/e02/model.pt\n", encoding="utf-8")
    assert load_config(cfg).checkpoint == "checkpoints/public/e02/model.pt"
    assert load_config("configs/app.yaml").checkpoint is None
