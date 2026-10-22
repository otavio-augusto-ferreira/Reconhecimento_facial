"""Fine-tuning: perdas, amostragem, aumento de dados, congelamento de camadas e uma época de treino."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from facerec.training.data import augment, limit_per_identity, pk_batches, random_batches
from facerec.training.losses import CosineClassifier, batch_hard_triplet_loss
from facerec.training.model import embed_crops, load_facenet, set_train_mode, set_trainable
from facerec.training.trainer import make_optimizer, train_one_epoch


# ------------------------------------------------------------------ perdas
def test_classificador_por_cosseno_sem_margem():
    head = CosineClassifier(8, 5, scale=10.0, margin=0.0)
    logits = head(torch.randn(4, 8), torch.tensor([0, 1, 2, 3]))
    assert logits.shape == (4, 5) and logits.abs().max() <= 10.0 + 1e-4


def test_arcface_reduz_so_o_logit_da_classe_correta():
    torch.manual_seed(0)
    emb, labels = torch.randn(6, 8), torch.tensor([0, 1, 2, 3, 4, 0])
    plain = CosineClassifier(8, 5, scale=30.0, margin=0.0)
    arc = CosineClassifier(8, 5, scale=30.0, margin=0.3)
    arc.weight.data.copy_(plain.weight.data)
    a, b = plain(emb, labels), arc(emb, labels)
    target = F.one_hot(labels, 5).bool()
    assert torch.all(b[target] < a[target])  # a margem torna a classe correta mais difícil
    assert torch.allclose(b[~target], a[~target])  # as outras ficam iguais
    assert torch.allclose(arc(emb), plain(emb))  # sem rótulos (inferência) não há margem


def test_prototipos_deixam_o_classificador_certo_desde_o_inicio():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 16)) * 5
    labels = np.repeat(np.arange(4), 10)
    emb = torch.from_numpy(centers[labels] + rng.normal(size=(40, 16))).float()
    head = CosineClassifier(16, 4)
    head.init_from_prototypes(emb, torch.from_numpy(labels))
    assert (head(emb).argmax(1).numpy() == labels).all()


def test_triplet_zero_com_classes_separadas_e_positivo_com_classes_misturadas():
    labels = torch.tensor([0, 0, 1, 1])
    sep = torch.tensor([[1.0, 0], [0.99, 0.01], [-1.0, 0], [-0.99, 0.01]])
    assert batch_hard_triplet_loss(sep, labels, margin=0.2).item() == pytest.approx(0.0, abs=1e-6)
    mixed = torch.tensor([[1.0, 0], [-1.0, 0], [0.99, 0.01], [-0.99, 0.01]])  # mesma pessoa, vetores opostos
    assert batch_hard_triplet_loss(mixed, labels, margin=0.2).item() > 1.0


def test_triplet_sem_ancora_valida_devolve_zero_e_gradiente_finito():
    emb = torch.randn(3, 4, requires_grad=True)
    loss = batch_hard_triplet_loss(emb, torch.tensor([0, 1, 2]))  # ninguém tem positivo
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(emb.grad).all()
    emb2 = torch.randn(6, 4, requires_grad=True)
    batch_hard_triplet_loss(emb2, torch.tensor([0, 0, 1, 1, 2, 2])).backward()
    assert torch.isfinite(emb2.grad).all()


# ------------------------------------------------------------------ dados
def test_limite_por_identidade_e_determinismo():
    items = [(f"a{i}.jpg", "A") for i in range(50)] + [(f"b{i}.jpg", "B") for i in range(3)]
    out = limit_per_identity(items, 10, seed=1)
    assert sum(1 for _, i in out if i == "A") == 10 and sum(1 for _, i in out if i == "B") == 3
    assert out == limit_per_identity(items, 10, seed=1)
    assert limit_per_identity(items, None) == items


def test_lotes_aleatorios_sao_completos_e_sem_repeticao():
    batches = list(random_batches(10, 4, np.random.default_rng(0)))
    assert len(batches) == 2 and all(len(b) == 4 for b in batches)
    assert len(set(np.concatenate(batches))) == 8


def test_lotes_pk_tem_p_identidades_com_k_imagens():
    labels = np.repeat(np.arange(6), 5)
    for batch in pk_batches(labels, p=3, k=4, steps=5, rng=np.random.default_rng(0)):
        assert len(batch) == 12
        ids, counts = np.unique(labels[batch], return_counts=True)
        assert len(ids) == 3 and (counts == 4).all()


def test_lotes_pk_reamostram_identidades_com_poucas_imagens_e_exigem_duas_identidades():
    labels = np.array([0, 0, 1, 1, 2])  # identidade 2 tem 1 imagem: é ignorada
    batch = next(iter(pk_batches(labels, p=2, k=4, steps=1, rng=np.random.default_rng(0))))
    assert len(batch) == 8 and 4 not in batch
    with pytest.raises(ValueError):
        next(iter(pk_batches(np.array([0, 0, 1]), p=2, k=2, steps=1, rng=np.random.default_rng(0))))


def test_aumento_mantem_formato_e_faixa_e_e_reprodutivel():
    x = torch.rand(6, 3, 32, 32) * 255
    a = augment(x, torch.Generator().manual_seed(1))
    b = augment(x, torch.Generator().manual_seed(1))
    c = augment(x, torch.Generator().manual_seed(2))
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 255
    assert torch.equal(a, b) and not torch.equal(a, c)


# ------------------------------------------------------------------ camadas congeladas
def test_modos_de_treino_liberam_so_os_blocos_certos():
    model = load_facenet(None)
    n_last = set_trainable(model, "last_block")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.split(".")[0] in {"repeat_3", "block8", "last_linear", "last_bn"} for n in trainable)
    assert n_last < sum(p.numel() for p in model.parameters())
    assert set_trainable(model, "full") == sum(p.numel() for p in model.parameters())
    with pytest.raises(ValueError):
        set_trainable(model, "inexistente")


def test_blocos_congelados_ficam_em_eval():
    model = load_facenet(None)
    set_trainable(model, "last_block")
    set_train_mode(model, "last_block")
    assert not model.conv2d_1a.bn.training and not model.repeat_1.training  # congelados
    assert model.repeat_3.training and model.last_bn.training  # treináveis


def _toy_data(n_classes=4, per_class=4, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), per_class)
    crops = rng.integers(0, 256, size=(len(labels), 160, 160, 3), dtype=np.uint8)
    return crops, labels


@pytest.mark.parametrize("loss_name", ["ce", "triplet"])
def test_uma_epoca_treina_so_o_que_deve(loss_name):
    torch.manual_seed(0)
    model = load_facenet(None)
    set_trainable(model, "last_block")
    crops, labels = _toy_data()
    head = CosineClassifier(512, 4, 30.0, 0.0) if loss_name == "ce" else None
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = make_optimizer(model, head, lr_backbone=1e-3, lr_head=1e-3, weight_decay=0.0)
    loss = train_one_epoch(
        model, head, loss_name, crops, labels, mode="last_block", optimizer=opt, batch_size=8,
        p=2, k=4, steps=2, triplet_margin=0.2, use_augment=True,
        rng=np.random.default_rng(0), gen=torch.Generator().manual_seed(0), device="cpu",
    )
    assert np.isfinite(loss)
    after = model.state_dict()
    frozen_same = all(torch.equal(before[k], after[k]) for k in before if not k.startswith(("repeat_3", "block8", "last_")))
    assert frozen_same  # inclusive as estatísticas do BatchNorm dos blocos congelados
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("block8") and "weight" in k)


def test_embeddings_sao_normalizados_e_nao_dependem_do_tamanho_do_lote():
    model = load_facenet(None)
    crops, _ = _toy_data(n_classes=2, per_class=3)
    a = embed_crops(model, crops, "cpu", batch_size=2)
    b = embed_crops(model, crops, "cpu", batch_size=6)
    assert a.shape == (6, 512) and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, b, atol=1e-4)


def test_prototipos_aceitam_embeddings_em_outro_dispositivo():
    """No Colab o classificador está na GPU e os embeddings iniciais chegam da CPU: o dispositivo do classificador manda."""
    head = CosineClassifier(4, 2)
    emb = torch.tensor([[1.0, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1.0, 0, 0]])
    head.init_from_prototypes(emb, torch.tensor([0, 0, 1]))
    assert head.weight.device == emb.device
    assert torch.allclose(head.weight.norm(dim=1), torch.ones(2), atol=1e-5)
