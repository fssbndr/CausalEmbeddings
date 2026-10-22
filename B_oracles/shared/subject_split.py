# Three-way subject split (train/val/test = 70/15/15) of each LARGE trial's baseline oracle.
# Embeddings are fit on train only (C_embeddings.py ood_fit_sources) and scored on the OOD path of
# D_evaluation.py: subject_val -> diagnostics, subject_test -> PEHE (E_diagnostic_validity.py joins them).
# Same T/Y/tau; only unit roles change. Both variants share one train fit: run C_embeddings.py on
# subject_val and symlink the embedding dirs for subject_test.
#
#   .venv/bin/python B_oracles/shared/subject_split.py rctbench_trial3 rctbench_trial4 ...
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

from knobs import resolve_out_dir  # same dir as this script, so it is on sys.path

TRAIN, VAL, SEED, MIN_N = 0.7, 0.15, 0, 2000  # test = remainder; MIN_N = config.CUDA_MIN_N

for trial in sys.argv[1:]:
    src = Path(resolve_out_dir(trial, "baseline"))
    cp = json.loads((src / "causal_params.json").read_text())
    oracle = pl.read_parquet(src / "semi_synthetic.parquet")
    ids = np.sort(oracle["ID"].unique().to_numpy())
    if cp.get("x_varies_by_seed") or len(ids) < MIN_N:  # no OOD path (IHDP, NEWS) / too few units
        print(f"skip {trial}")
        continue
    perm = np.random.default_rng(SEED).permutation(ids).tolist()
    a, b = int(len(ids) * TRAIN), int(len(ids) * (TRAIN + VAL))
    role = {i: "train" for i in perm[:a]} | {i: "val" for i in perm[a:b]} | {i: "test" for i in perm[b:]}
    labelled = oracle.with_columns(pl.col("ID").replace_strict(role).alias("source_dataset"))
    for name, ood in {"subject_val": ["val"], "subject_test": ["test"]}.items():
        dst = Path(resolve_out_dir(trial, name))
        dst.mkdir(parents=True, exist_ok=True)
        labelled.write_parquet(dst / "semi_synthetic.parquet")
        shutil.copy(src / "feature_cols.json", dst / "feature_cols.json")
        (dst / "causal_params.json").write_text(json.dumps(cp | {"ood_sources": ood, "ood_fit_sources": ["train"]}, indent=2))
        print(f"{trial}: {name} -> {dst}")
