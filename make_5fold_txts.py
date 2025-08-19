#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：make_5fold_txts.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/8/19 上午9:42 
"""
import os
import argparse
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def read_labeled_txt(path: str) -> List[Tuple[str, int, int]]:
    """Lines: path [main] [aux].
    If aux is missing, fill with -1.
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split(',')
            if len(toks) < 2:
                continue
            img = toks[0]
            if toks[1] == '良性':
                main = 0
            else:
                main = 1
            if len(toks) > 2:
                if toks[2] == '非肿大':
                    aux = 0
                else:
                    aux = 1
            else:
                aux = -1
            items.append((img, main, aux))
    return items


def read_weak_txt(path: str, threshold: float = 0.5) -> List[Tuple[str, int]]:
    """Lines could be:
    - path aux_label
    - path aux_prob  (float in [0,1]) -> binarize by threshold
    - path aux_label aux_prob (we take aux_label if int, else prob)
    Returns (path, aux_label_int in {0,1}).
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split()
            if len(toks) < 2:
                continue
            img = toks[0]
            aux = toks[1]
            # try parse as float prob first
            try:
                val = float(aux)
                aux_int = 1 if val >= threshold else 0
            except ValueError:
                # not a float, expect int label
                aux_int = int(aux)
            items.append((img, aux_int))
    return items


def write_triplets(path: str, triplets: List[Tuple[str, int, int]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for p, m, a in triplets:
            f.write(f"{p},{m},{a}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labeled_txt', help='20250702-良恶性2分类-all.txt (path main [aux])')
    ap.add_argument('--weak_txt', help='20250812-肿大软标签.txt (path aux or path prob)')
    ap.add_argument('--out_dir', help='./')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--prob_threshold', type=float, default=0.5,
                    help='threshold if weak txt stores probability')
    ap.add_argument('--val_offset', type=int, default=1,
                    help='which fold to use as val relative to test fold (test=i, val=(i+offset)%K)')
    ap.add_argument('--exclude_aux_overlap', choices=['none','valtest','all_labeled'], default='valtest',
                    help='drop aux-only samples that overlap with labeled val/test (default) or with any labeled sample')
    args = ap.parse_args()

    args.labeled_txt = r'E:\med_dataset\lymph淋巴结\中山淋巴结\训练集txt\ori\20250702-良恶性2分类-all.txt'
    args.weak_txt = r'E:\med_dataset\lymph淋巴结\中山淋巴结\训练集txt\ori\20250812-肿大软标签.txt'
    args.out_dir = r'E:\med_dataset\lymph淋巴结\中山淋巴结\训练集txt\ori'
    labeled = read_labeled_txt(args.labeled_txt)
    weak = read_weak_txt(args.weak_txt, threshold=args.prob_threshold)

    imgs = [p for p, m, a in labeled]
    y = [m for p, m, a in labeled]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.arange(len(imgs)), y))

    # For quick lookup
    weak_dict = {p: aux for p, aux in weak}

    # Optional global labeled path set
    all_labeled_paths = set(imgs)

    summary = []

    for i in range(args.folds):
        test_idx = splits[i][1]
        val_idx = splits[(i + args.val_offset) % args.folds][1]
        train_idx = np.setdiff1d(np.arange(len(imgs)), np.union1d(test_idx, val_idx))

        def subset(idxs):
            return [labeled[k] for k in idxs]

        train_l = subset(train_idx)
        val_l = subset(val_idx)
        test_l = subset(test_idx)

        # Filter weak -> aux-only for this fold
        banned = set(p for p, _, _ in val_l) | set(p for p, _, _ in test_l)
        if args.exclude_aux_overlap == 'all_labeled':
            banned |= all_labeled_paths

        aux_train = []
        for p, aux in weak:
            if p in banned:
                continue
            # produce triplet: main=-1, aux=aux
            aux_train.append((p, -1, int(aux)))

        out_fold = os.path.join(args.out_dir, f'fold{i}')
        os.makedirs(out_fold, exist_ok=True)

        write_triplets(os.path.join(out_fold, 'train_labeled.txt'), train_l)
        write_triplets(os.path.join(out_fold, 'val.txt'), val_l)
        write_triplets(os.path.join(out_fold, 'test.txt'), test_l)
        write_triplets(os.path.join(out_fold, 'train_auxonly.txt'), aux_train)

        summary.append({
            'fold': i,
            'train_labeled': len(train_l),
            'val': len(val_l),
            'test': len(test_l),
            'train_auxonly': len(aux_train),
        })

    # Print summary
    print('Done. Summary:')
    for s in summary:
        print(s)


if __name__ == '__main__':
    main()
