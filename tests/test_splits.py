"""Testes da spec 006 (docs/specs/006-dataset-splits.md). Usam apenas dados sintéticos."""
import pytest

from facerec.data.splits import (
    LeakageError,
    check_no_leakage,
    filter_min_images,
    find_cross_split_duplicates,
    load_splits,
    make_verification_pairs,
    save_splits,
    scan_identity_folders,
    split_by_identity,
)


def synthetic_items(n_ids=20, imgs_per_id=4):
    return [(f"root/p{i:02d}/img{k}.jpg", f"p{i:02d}") for i in range(n_ids) for k in range(imgs_per_id)]


def identities(items):
    return {i for _, i in items}


def test_ca1_no_identity_in_more_than_one_split():
    splits = split_by_identity(synthetic_items())
    assert identities(splits["train"]).isdisjoint(identities(splits["val"]))
    assert identities(splits["train"]).isdisjoint(identities(splits["test"]))
    assert identities(splits["val"]).isdisjoint(identities(splits["test"]))
    check_no_leakage(splits, check_hashes=False)


def test_all_items_are_kept_once():
    items = synthetic_items()
    splits = split_by_identity(items)
    merged = [it for subset in splits.values() for it in subset]
    assert sorted(merged) == sorted(items)


def test_ca2_same_seed_same_split_and_different_seed_differs():
    items = synthetic_items()
    assert split_by_identity(items, seed=1) == split_by_identity(items, seed=1)
    assert split_by_identity(items, seed=1) != split_by_identity(items, seed=2)


def test_ca3_pairs_are_balanced_and_correct():
    items = synthetic_items()
    owner = {p: i for p, i in items}
    pairs = make_verification_pairs(items, n_pairs=100, seed=0)
    labels = [y for _, _, y in pairs]
    assert labels.count(1) == labels.count(0) == 50
    for a, b, y in pairs:
        assert a != b
        assert (owner[a] == owner[b]) == (y == 1)


def test_ca4_leakage_check_detects_shared_identity():
    splits = split_by_identity(synthetic_items())
    shared = splits["train"][0]
    splits["test"].append(("root/other/x.jpg", shared[1]))
    with pytest.raises(LeakageError):
        check_no_leakage(splits, check_hashes=False)


def test_ca4_leakage_check_detects_identical_files(tmp_path):
    a = tmp_path / "alice" / "1.jpg"
    b = tmp_path / "bob" / "1.jpg"
    for f in (a, b):
        f.parent.mkdir()
        f.write_bytes(b"mesmo conteudo")
    splits = {"train": [(str(a), "alice")], "test": [(str(b), "bob")]}
    assert find_cross_split_duplicates(splits)
    with pytest.raises(LeakageError):
        check_no_leakage(splits)


def test_ca5_identities_with_single_image_are_dropped():
    items = synthetic_items(n_ids=5, imgs_per_id=3) + [("root/solo/a.jpg", "solo")]
    assert "solo" not in identities(filter_min_images(items, min_images=2))


def test_max_images_caps_dominant_identities():
    items = synthetic_items(n_ids=3, imgs_per_id=10)
    capped = filter_min_images(items, min_images=2, max_images=4)
    assert all(sum(1 for _, i in capped if i == ident) == 4 for ident in identities(capped))


def test_scan_and_manifest_roundtrip(tmp_path):
    for person in ("ana", "bia", "caio", "duda"):
        d = tmp_path / person
        d.mkdir()
        for k in range(3):
            (d / f"{person}_{k}.jpg").write_bytes(f"{person}{k}".encode())
    items = scan_identity_folders(tmp_path)
    assert len(items) == 12
    splits = split_by_identity(items, seed=7)
    check_no_leakage(splits)
    manifest = tmp_path.parent / f"{tmp_path.name}_manifest.json"
    save_splits(splits, manifest, tmp_path, seed=7)
    assert load_splits(manifest, tmp_path) == splits


def test_invalid_ratios_and_too_few_identities():
    with pytest.raises(ValueError):
        split_by_identity(synthetic_items(), ratios=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError):
        split_by_identity(synthetic_items(n_ids=2))


def test_manifest_with_windows_backslashes_loads_on_any_system(tmp_path):
    """Manifestos criados no Windows (com "\\") precisam funcionar no Linux/Colab."""
    import json

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"seed": 1, "splits": {"train": [["Ana\\Ana_0001.jpg", "Ana"]], "val": [["Bia/Bia_0001.jpg", "Bia"]]}}),
        encoding="utf-8",
    )
    splits = load_splits(manifest, tmp_path)
    assert splits["train"][0][0] == str(tmp_path / "Ana/Ana_0001.jpg")
    assert splits["val"][0][0] == str(tmp_path / "Bia/Bia_0001.jpg")


def test_saved_manifest_uses_forward_slashes(tmp_path):
    import json

    img = tmp_path / "Ana" / "Ana_0001.jpg"
    img.parent.mkdir()
    img.write_bytes(b"x")
    manifest = tmp_path / "m.json"
    save_splits({"train": [(str(img), "Ana")]}, manifest, tmp_path, seed=1)
    assert json.loads(manifest.read_text(encoding="utf-8"))["splits"]["train"][0][0] == "Ana/Ana_0001.jpg"
