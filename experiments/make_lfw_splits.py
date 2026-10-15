"""Cria a divisão treino/validação/teste do LFW por identidade.

Uso (na raiz do repositório, após `pip install -e .`):
    python experiments/make_lfw_splits.py --root data/public/lfw \
        --out experiments/splits/lfw_seed42.json --seed 42

O manifesto gerado contém apenas nomes de arquivos do dataset público (sem imagens).
"""
import argparse

from facerec.data.splits import (
    check_no_leakage,
    filter_min_images,
    save_splits,
    scan_identity_folders,
    split_by_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="pasta do LFW (uma subpasta por pessoa)")
    parser.add_argument("--out", default="experiments/splits/lfw_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    items = scan_identity_folders(args.root)
    if not items:
        raise SystemExit(f"nenhuma imagem encontrada em {args.root}")
    items = filter_min_images(items, args.min_images, args.max_images, args.seed)
    splits = split_by_identity(items, seed=args.seed)
    check_no_leakage(splits)
    save_splits(splits, args.out, args.root, args.seed)

    for name, subset in splits.items():
        print(f"{name}: {len({i for _, i in subset})} identidades, {len(subset)} imagens")
    print(f"manifesto salvo em {args.out} (sem leakage de identidade, caminho ou hash)")


if __name__ == "__main__":
    main()
