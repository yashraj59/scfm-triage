"""
Foundation Model Triage
=======================

Run a practical zero-shot benchmark on your own single-cell data before
choosing a foundation-model base to invest in.

The script takes a labeled AnnData file, downloads public model assets when
needed, extracts embeddings from Geneformer V2, scGPT, and scFoundation, and
compares the bases on decision signals that matter for labeled single-cell
datasets, including optional time-course and focal-cluster analyses.

Three evaluations:
  (a) Linear probe cell type classification (5-fold stratified)
  (b) Diffusion pseudotime recovery (Spearman vs true time)
  (c) Cluster separation between a user-selected pair of focal labels

Usage:
    python foundation_model_triage.py \\
        --adata /path/to/eval_set.h5ad \\
        --qc-done \\
        --cell-type-col cell_type \\
        --time-col time_point \\
        --focal-labels iPSC,partial_reprog \\
        --output ./triage_results/

The script bootstraps an isolated virtual environment by default:
    ./.foundation_model_triage_env

It installs the analysis stack, downloads/reuses public model assets under
./public_models, and can be run with only your AnnData path plus metadata
column names. The default focal-label pair is useful for iPSC reprogramming
data, but any two labels can be supplied. Use --setup-only to create/update the
env without running.
"""

from __future__ import annotations

import os
import json
import argparse
import subprocess
import sys
import tempfile
import venv
import warnings
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple


BOOTSTRAP_ENV_VAR = "SCFM_TRIAGE_BOOTSTRAPPED"
DEFAULT_ENV_DIR = ".foundation_model_triage_env"

BASE_REQUIREMENTS = [
    "numpy<2",
    "pandas",
    "scanpy",
    "anndata",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "h5py",
    "packaging",
    "huggingface-hub",
]

MODEL_REQUIREMENTS = {
    "geneformer": [
        "torch==2.3.0",
        "transformers<5",
        "datasets",
        "loompy",
        "git+https://github.com/lcrawlab/Geneformer.git",
    ],
    "scgpt": [
        "IPython",
        "torch==2.3.0",
        "scgpt==0.2.4",
    ],
    "scfoundation": [
        "torch==2.3.0",
        "einops",
        "local-attention==1.9.0",
        "tqdm",
    ],
}


def _venv_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _bootstrap_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--env-dir", default=DEFAULT_ENV_DIR)
    p.add_argument("--no-auto-env", action="store_true")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--geneformer-model", default=None)
    p.add_argument("--scgpt-model", default=None)
    p.add_argument("--scfoundation-repo", default=None)
    p.add_argument("--scfoundation-ckpt", default=None)
    p.add_argument("--model-cache", default="public_models")
    return p


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Zero-shot triage for Geneformer V2, scGPT, and scFoundation on "
            "your labeled single-cell AnnData."
        )
    )
    p.add_argument("--adata", required=True, help="Path to evaluation AnnData (.h5ad)")
    p.add_argument("--output", default="./triage_results")
    p.add_argument("--env-dir", default=DEFAULT_ENV_DIR,
                   help="Virtualenv directory used for automatic setup")
    p.add_argument("--model-cache", default="public_models",
                   help="Directory where public model assets are downloaded/reused")
    p.add_argument("--no-auto-env", action="store_true",
                   help="Disable automatic virtualenv setup/re-exec")
    p.add_argument("--setup-only", action="store_true",
                   help="Create/update the virtualenv, then exit")
    p.add_argument("--n-cells", type=int, default=100_000,
                   help="Maximum cells to evaluate after optional QC; larger datasets are stratified-subsampled")
    qc_group = p.add_mutually_exclusive_group(required=True)
    qc_group.add_argument("--qc-done", action="store_true",
                          help="Input AnnData is already QC-filtered; skip built-in QC filters")
    qc_group.add_argument("--run-qc", action="store_true",
                          help="Run built-in basic QC filters before evaluation")
    p.add_argument("--cell-type-col", default="cell_type")
    p.add_argument("--time-col", default="time_point")
    p.add_argument("--gene-col", default="feature_name")
    p.add_argument("--focal-labels", default="iPSC,partial_reprog",
                   help="Comma-separated pair for cluster separation eval; override for non-iPSC datasets")
    p.add_argument("--ensembl-map", default=None,
                   help="CSV with columns 'symbol','ensembl_id' (for Geneformer)")
    p.add_argument("--geneformer-model", default=None,
                   help="Optional local Geneformer model path; defaults to public V2 download")
    p.add_argument("--scgpt-model", default=None,
                   help="Optional local scGPT model path; defaults to public human checkpoint")
    p.add_argument("--scfoundation-repo", default=None,
                   help="Optional local scFoundation repo path; defaults to public GitHub clone")
    p.add_argument("--scfoundation-ckpt", default=None,
                   help="Optional local scFoundation checkpoint; defaults to public HF checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Re-extract embeddings even if cached")
    return p


def parse_args():
    return build_arg_parser().parse_args()


def _all_models_requested_by_default(args: argparse.Namespace) -> bool:
    return not (
        args.geneformer_model
        or args.scgpt_model
        or args.scfoundation_repo
        or args.scfoundation_ckpt
    )


def _requirements_for_args(args: argparse.Namespace) -> list[str]:
    requirements = list(BASE_REQUIREMENTS)
    all_default = _all_models_requested_by_default(args)
    if all_default or args.geneformer_model:
        requirements.extend(MODEL_REQUIREMENTS["geneformer"])
    if all_default or args.scgpt_model:
        requirements.extend(MODEL_REQUIREMENTS["scgpt"])
    if all_default or args.scfoundation_repo or args.scfoundation_ckpt:
        requirements.extend(MODEL_REQUIREMENTS["scfoundation"])
    return sorted(dict.fromkeys(requirements))


def _install_requirements(env_python: Path, requirements: list[str]) -> None:
    """Install in small phases so ML packages can see torch during setup."""
    install_groups: list[list[str]] = [BASE_REQUIREMENTS]
    if "torch==2.3.0" in requirements:
        install_groups.append(["torch==2.3.0"])
    if "git+https://github.com/lcrawlab/Geneformer.git" in requirements:
        install_groups.append([
            "transformers<5",
            "datasets",
            "loompy",
            "git+https://github.com/lcrawlab/Geneformer.git",
        ])
    if "scgpt==0.2.4" in requirements:
        install_groups.append(["IPython", "scgpt==0.2.4"])
    scfoundation_reqs = [
        req for req in ("einops", "local-attention==1.9.0", "tqdm") if req in requirements
    ]
    if scfoundation_reqs:
        install_groups.append([req for req in scfoundation_reqs if req != "local-attention==1.9.0"])

    seen: set[str] = set()
    for group in install_groups:
        group = [req for req in group if req in requirements and req not in seen]
        if not group:
            continue
        seen.update(group)
        subprocess.check_call([str(env_python), "-m", "pip", "install", *group])

    if "local-attention==1.9.0" in requirements and "local-attention==1.9.0" not in seen:
        subprocess.check_call([
            str(env_python),
            "-m",
            "pip",
            "install",
            "local-attention==1.9.0",
            "fsspec==2024.6.1",
        ])


def ensure_runtime_environment() -> None:
    """Create/update a local venv and re-exec the script inside it."""
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        return
    boot_args, _ = _bootstrap_parser().parse_known_args()
    if boot_args.no_auto_env or os.environ.get(BOOTSTRAP_ENV_VAR) == "1":
        return

    env_dir = Path(boot_args.env_dir).expanduser()
    if not env_dir.is_absolute():
        env_dir = Path.cwd() / env_dir
    env_python = _venv_python(env_dir)

    if Path(sys.executable).resolve() != env_python.resolve():
        if not env_python.exists():
            print(f"[env] Creating virtual environment: {env_dir}", flush=True)
            venv.create(env_dir, with_pip=True, clear=False)

        requirements = _requirements_for_args(boot_args)
        marker = env_dir / ".foundation_model_triage_requirements.txt"
        wanted = "\n".join(requirements) + "\n"
        if not marker.exists() or marker.read_text() != wanted:
            print("[env] Installing/updating packages:", flush=True)
            for req in requirements:
                print(f"  - {req}", flush=True)
            subprocess.check_call([
                str(env_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "wheel",
                "setuptools",
            ])
            _install_requirements(env_python, requirements)
            marker.write_text(wanted)
        else:
            print(f"[env] Reusing {env_dir}", flush=True)

        env = os.environ.copy()
        env[BOOTSTRAP_ENV_VAR] = "1"
        if boot_args.setup_only:
            print(f"[env] Setup complete. Python: {env_python}", flush=True)
            raise SystemExit(0)
        print(f"[env] Restarting inside {env_python}", flush=True)
        os.execve(str(env_python), [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


ensure_runtime_environment()
if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parse_args()
    raise SystemExit(0)

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, silhouette_score
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_eval_set(
    adata_path: str,
    n_cells: int = 100_000,
    cell_type_col: str = "cell_type",
    time_col: Optional[str] = "time_point",
    qc_done: bool = False,
    min_genes: int = 500,
    max_mt_pct: float = 20.0,
    seed: int = 42,
) -> ad.AnnData:
    """
    Load and prepare evaluation set with stratified subsampling.

    Expects an AnnData with:
      - .X = raw counts (UMI counts; will be normalized only for some models)
      - .var index = gene symbols (we'll map to ENSEMBL for Geneformer)
      - .obs[cell_type_col] = ground truth cell type labels
      - .obs[time_col] = time point (numeric, for pseudotime eval; can be NaN)
    """
    print(f"Loading {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    print(f"  Raw: {adata.n_obs} cells x {adata.n_vars} genes")

    if qc_done:
        print("  QC: skipped (--qc-done)")
    else:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=10)
        adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], inplace=True, log1p=False, percent_top=None
        )
        adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct].copy()
        print(f"  After QC: {adata.n_obs} cells")

    # Stratified subsample by cell type to keep classes balanced
    if adata.n_obs > n_cells:
        rng = np.random.default_rng(seed)
        keep_idx = []
        for ct, group in adata.obs.groupby(cell_type_col):
            n_take = max(
                50,  # floor per class
                min(len(group), int(n_cells * len(group) / len(adata.obs)))
            )
            keep_idx.extend(
                rng.choice(group.index.values, size=min(n_take, len(group)),
                           replace=False).tolist()
            )
        adata = adata[keep_idx].copy()
        print(f"  Stratified subsample: {adata.n_obs} cells")

    # Store raw counts; we'll need them in different normalizations per model
    adata.layers["counts"] = adata.X.copy()
    return adata


# ============================================================================
# EMBEDDING EXTRACTORS
# ============================================================================

def get_geneformer_embeddings(
    adata: ad.AnnData,
    model_dir: str,
    ensembl_map: Optional[Dict[str, str]] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract Geneformer V2 cell embeddings.

    Geneformer rank-encodes genes by normalized expression and returns
    contextualized embeddings via the transformer. We mean-pool the last
    hidden state across tokens (excluding pad) to get a cell embedding.

    Note: Geneformer expects Ensembl IDs. If your var index has gene symbols,
    pass `ensembl_map` (dict: symbol -> ensembl_id).
    """
    from geneformer import TranscriptomeTokenizer, EmbExtractor
    import inspect

    print("[Geneformer] Preparing input...")
    work_dir = tempfile.mkdtemp(prefix="geneformer_")
    input_dir = Path(work_dir) / "input"
    input_dir.mkdir()

    adata_gf = adata.copy()
    # Geneformer needs ensembl_id in var and n_counts in obs
    if ensembl_map is not None:
        adata_gf.var["ensembl_id"] = adata_gf.var_names.map(ensembl_map)
        adata_gf = adata_gf[:, adata_gf.var["ensembl_id"].notna()].copy()
    else:
        # Assume var_names already are Ensembl IDs
        adata_gf.var["ensembl_id"] = adata_gf.var_names

    counts = adata_gf.layers["counts"]
    adata_gf.obs["n_counts"] = (
        counts.sum(axis=1).A1 if hasattr(counts, "A1") else counts.sum(axis=1)
    )

    # Geneformer needs a unique cell ID
    adata_gf.obs["cell_id"] = np.arange(adata_gf.n_obs).astype(str)

    adata_gf.write_h5ad(input_dir / "data.h5ad")

    print("[Geneformer] Tokenizing...")
    supported = inspect.signature(TranscriptomeTokenizer).parameters
    dictionary_dir = Path(model_dir).resolve().parent / "geneformer"
    token_dict_path = dictionary_dir / "token_dictionary_gc104M.pkl"
    gene_median_path = dictionary_dir / "gene_median_dictionary_gc104M.pkl"
    if not token_dict_path.exists() and "token_dictionary_file" in supported:
        token_dict_path = Path(supported["token_dictionary_file"].default)
    if not gene_median_path.exists() and "gene_median_file" in supported:
        gene_median_path = Path(supported["gene_median_file"].default)

    has_cls_sep = False
    if token_dict_path and token_dict_path.exists():
        token_dict = pickle.load(open(token_dict_path, "rb"))
        if "<sep>" not in token_dict and "<eos>" in token_dict:
            token_dict = dict(token_dict)
            token_dict["<sep>"] = token_dict["<eos>"]
            token_dict_path = Path(work_dir) / "token_dictionary_compat.pkl"
            with open(token_dict_path, "wb") as f:
                pickle.dump(token_dict, f)
        has_cls_sep = "<cls>" in token_dict and "<sep>" in token_dict

    tokenizer_kwargs = {
        "custom_attr_name_dict": {
            "cell_id": "cell_id",
            "n_counts": "n_counts",
        },
        "nproc": 4,
        "model_input_size": 4096,  # V2 uses 4096; V1 used 2048
        "special_token": has_cls_sep,
        "collapse_gene_ids": True,
        "gene_median_file": gene_median_path,
        "token_dictionary_file": token_dict_path,
    }
    tokenizer_kwargs = {
        key: value for key, value in tokenizer_kwargs.items() if key in supported
    }
    tokenizer = TranscriptomeTokenizer(**tokenizer_kwargs)
    try:
        tokenizer.tokenize_data(
            data_directory=str(input_dir),
            output_directory=work_dir,
            output_prefix="tokenized",
            file_format="h5ad",
        )
    except AttributeError as exc:
        if "'str' object has no attribute 'var'" not in str(exc):
            raise
        tokenizer.tokenize_data(
            data_directory=adata_gf,
            output_directory=work_dir,
            output_prefix="tokenized",
            file_format="h5ad",
        )

    print("[Geneformer] Extracting embeddings...")
    extractor = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        max_ncells=None,
        emb_layer=-1,
        forward_batch_size=batch_size,
        nproc=4,
    )
    emb_df = extractor.extract_embs(
        model_directory=model_dir,
        input_data_file=str(Path(work_dir) / "tokenized.dataset"),
        output_directory=work_dir,
        output_prefix="emb",
    )

    # emb_df is a DataFrame indexed by cell with embedding dims as columns
    # Re-align to original adata order via cell_id
    emb_df = emb_df.set_index("cell_id") if "cell_id" in emb_df.columns else emb_df
    cell_ids = adata_gf.obs["cell_id"].values
    if set(cell_ids).issubset(set(emb_df.index.astype(str))):
        emb_df.index = emb_df.index.astype(str)
        embeddings = emb_df.loc[cell_ids].values.astype(np.float32)
    elif len(emb_df) == adata_gf.n_obs:
        embeddings = emb_df.values.astype(np.float32)
    else:
        raise ValueError(
            "Geneformer embeddings could not be aligned to cells: "
            f"{len(emb_df)} embeddings for {adata_gf.n_obs} cells."
        )

    print(f"[Geneformer] Embeddings: {embeddings.shape}")
    return embeddings


def get_scgpt_embeddings(
    adata: ad.AnnData,
    model_dir: str,
    gene_col: str = "feature_name",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract scGPT cell embeddings using their zero-shot embedding API.

    scGPT uses value-aware binned encoding and gene symbols.
    """
    import scgpt as scg

    print("[scGPT] Preparing input...")
    adata_sc = adata.copy()
    # scGPT's embed_data normalizes internally if we pass raw counts in .X
    adata_sc.X = adata_sc.layers["counts"]

    # scGPT looks up genes via a column in .var; ensure it exists
    if gene_col not in adata_sc.var.columns:
        adata_sc.var[gene_col] = adata_sc.var_names

    print("[scGPT] Embedding cells...")
    adata_out = scg.tasks.embed_data(
        adata_sc,
        model_dir=model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        use_fast_transformer=False,
        return_new_adata=True,
        # Use HVG=False here so we evaluate the model's own gene selection priors
    )
    if "X_scGPT" in adata_out.obsm:
        embeddings = adata_out.obsm["X_scGPT"].astype(np.float32)
    else:
        embeddings = np.asarray(adata_out.X).astype(np.float32)
    print(f"[scGPT] Embeddings: {embeddings.shape}")
    return embeddings


def get_scfoundation_embeddings(
    adata: ad.AnnData,
    repo_dir: str,
    model_ckpt: str,
    output_type: str = "cell",
) -> np.ndarray:
    """
    Extract scFoundation embeddings.

    scFoundation has a separate inference script. The cleanest integration is:
      1. Write adata in their expected format
      2. Subprocess-call their get_embedding.py
      3. Load the resulting .npy

    See: https://github.com/biomap-research/scFoundation/tree/main/model
    """
    import subprocess

    print("[scFoundation] Preparing input...")
    work_dir = tempfile.mkdtemp(prefix="scfoundation_")
    work_path = Path(work_dir).resolve()
    input_csv = work_path / "input.csv"
    repo_model_dir = (Path(repo_dir).resolve() / "model")
    expected_ckpt = repo_model_dir / "models" / "models.ckpt"
    if model_ckpt:
        source_ckpt = Path(model_ckpt).expanduser().resolve()
        expected_ckpt.parent.mkdir(parents=True, exist_ok=True)
        if not expected_ckpt.exists():
            expected_ckpt.symlink_to(source_ckpt)

    # scFoundation expects a CSV: rows=cells, cols=genes (symbols)
    counts = adata.layers["counts"]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    df = pd.DataFrame(counts, index=adata.obs_names, columns=adata.var_names)
    df.to_csv(input_csv)

    print("[scFoundation] Running inference (this may take a while)...")
    cmd = [
        sys.executable, "get_embedding.py",
        "--task_name", "foundation_model_triage",
        "--input_type", "singlecell",
        "--output_type", output_type,
        "--pool_type", "all",
        "--tgthighres", "f1",
        "--data_path", str(input_csv),
        "--save_path", str(work_path),
        "--pre_normalized", "F",
        "--version", "ce",
        "--model_path", model_ckpt,
    ]
    result = subprocess.run(cmd, cwd=repo_model_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"scFoundation failed: {result.stderr}")

    # Their script writes <task_name>_..._embedding.npy
    npy_files = list(work_path.glob("*_embedding*.npy"))
    if not npy_files:
        raise FileNotFoundError("scFoundation did not produce embeddings")
    embeddings = np.load(npy_files[0]).astype(np.float32)
    print(f"[scFoundation] Embeddings: {embeddings.shape}")
    return embeddings


# ============================================================================
# EVALUATION 1: LINEAR PROBE
# ============================================================================

def linear_probe_eval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """5-fold stratified logistic regression probe on frozen embeddings."""
    X = StandardScaler().fit_transform(embeddings)
    y = pd.Categorical(labels).codes

    # Drop classes with <n_splits members (StratifiedKFold requirement)
    counts = pd.Series(y).value_counts()
    keep_classes = counts[counts >= n_splits].index.values
    keep_mask = np.isin(y, keep_classes)
    X, y = X[keep_mask], y[keep_mask]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s, baccs = [], [], []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(
            max_iter=2000, n_jobs=-1,
            multi_class="multinomial", C=1.0, solver="lbfgs",
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        baccs.append(balanced_accuracy_score(y[te], pred))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "balanced_acc_mean": float(np.mean(baccs)),
        "balanced_acc_std": float(np.std(baccs)),
        "n_classes": int(len(keep_classes)),
        "n_cells_evaluated": int(keep_mask.sum()),
    }


# ============================================================================
# EVALUATION 2: PSEUDOTIME RECOVERY
# ============================================================================

def pseudotime_eval(
    embeddings: np.ndarray,
    true_time: np.ndarray,
    n_neighbors: int = 30,
) -> Dict[str, float]:
    """
    Build a kNN graph in embedding space, compute diffusion pseudotime,
    return Spearman correlation with true time. Root = earliest time point.
    """
    valid = ~pd.isna(true_time)
    if valid.sum() < 200:
        return {"note": "insufficient time-labeled cells", "n_cells": int(valid.sum())}

    embs = embeddings[valid]
    times = np.asarray(true_time)[valid].astype(float)
    if len(np.unique(times)) < 2:
        return {
            "note": "requires at least two distinct time points",
            "n_cells": int(valid.sum()),
            "n_time_points": int(len(np.unique(times))),
        }

    adata_emb = ad.AnnData(X=embs)
    adata_emb.obs["true_time"] = times
    adata_emb.obsm["X_emb"] = embs

    sc.pp.neighbors(adata_emb, n_neighbors=n_neighbors, use_rep="X_emb")

    # Root cell: pick the cell at the earliest time point that is most central
    # (highest connectivity) — more robust than picking arbitrary first cell
    min_time = times.min()
    early_idx = np.where(times == min_time)[0]
    conn = np.asarray(adata_emb.obsp["connectivities"][early_idx].sum(axis=1)).flatten()
    root = early_idx[np.argmax(conn)]
    adata_emb.uns["iroot"] = int(root)

    try:
        sc.tl.diffmap(adata_emb)
        sc.tl.dpt(adata_emb)
    except Exception as e:
        return {"error": f"DPT failed: {e}"}

    ptime = adata_emb.obs["dpt_pseudotime"].values
    finite = np.isfinite(ptime)
    if finite.sum() < 100:
        return {"note": "DPT produced too few finite values"}

    rho, pval = spearmanr(ptime[finite], times[finite])
    return {
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "n_cells_evaluated": int(finite.sum()),
        "n_time_points": int(len(np.unique(times))),
    }


# ============================================================================
# EVALUATION 3: CLUSTER SEPARATION (iPSC vs partial reprog)
# ============================================================================

def cluster_separation_eval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    focal_labels: Tuple[str, str] = ("iPSC", "partial_reprog"),
) -> Dict[str, float]:
    """
    How well does the model separate fully-reprogrammed iPSC from
    partially-reprogrammed intermediates? Uses silhouette + centroid distance.
    """
    labels_arr = np.asarray(labels)
    mask = np.isin(labels_arr, focal_labels)
    if mask.sum() < 100:
        return {
            "note": f"insufficient cells in {focal_labels}",
            "n_cells_focal": int(mask.sum()),
            "available_labels": sorted(set(labels_arr))[:30],
        }

    sub_emb = embeddings[mask]
    sub_lab = labels_arr[mask]

    sil = silhouette_score(sub_emb, sub_lab, metric="cosine")

    c0 = sub_emb[sub_lab == focal_labels[0]].mean(axis=0)
    c1 = sub_emb[sub_lab == focal_labels[1]].mean(axis=0)
    centroid_cos = cosine(c0, c1)

    # Overlap: fraction of focal_labels[1] cells whose nearest neighbor
    # (in cosine) among focal_labels[0]+[1] is of focal_labels[0]
    # — proxy for confusion
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(sub_emb)
    _, idx = nn.kneighbors(sub_emb)
    nn_labels = sub_lab[idx[:, 1]]
    confusion_rate = float(np.mean(sub_lab != nn_labels))

    return {
        "silhouette_cosine": float(sil),
        "centroid_cosine_dist": float(centroid_cos),
        "nn_confusion_rate": confusion_rate,
        "n_cells_focal": int(mask.sum()),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def make_comparison_plot(results: Dict, output_path: str):
    """Bar plot comparing the three evals across models."""
    models = list(results.keys())
    metrics = {
        "Linear probe (bal. acc)": [
            results[m]["linear_probe"].get("balanced_acc_mean", 0) for m in models
        ],
        "Pseudotime (Spearman ρ)": [
            results[m]["pseudotime"].get("spearman_rho", 0) for m in models
        ],
        "Cluster sep. (silhouette)": [
            results[m]["cluster_separation"].get("silhouette_cosine", 0)
            for m in models
        ],
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, vals) in zip(axes, metrics.items()):
        sns.barplot(x=models, y=vals, ax=ax, palette="viridis")
        ax.set_title(name)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def _download_hf_snapshot(repo_id: str, local_dir: Path, allow_patterns=None) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
    )
    return local_dir


def ensure_public_model_assets(args) -> None:
    """Fill missing model paths by downloading public checkpoints/assets."""
    cache_dir = Path(args.model_cache).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = Path.cwd() / cache_dir

    run_all_by_default = not (
        args.geneformer_model
        or args.scgpt_model
        or args.scfoundation_repo
        or args.scfoundation_ckpt
    )

    if run_all_by_default:
        geneformer_root = cache_dir / "Geneformer"
        geneformer_model = geneformer_root / "Geneformer-V2-104M"
        if not (geneformer_model / "config.json").exists():
            print("[models] Downloading Geneformer V2 104M...")
            _download_hf_snapshot(
                "ctheodoris/Geneformer",
                geneformer_root,
                allow_patterns=[
                    "Geneformer-V2-104M/*",
                    "geneformer/*gc104M.pkl",
                ],
            )
        args.geneformer_model = str(geneformer_model)

    if run_all_by_default:
        scgpt_model = cache_dir / "scgpt-human"
        if not (scgpt_model / "best_model.pt").exists():
            print("[models] Downloading scGPT human checkpoint...")
            _download_hf_snapshot("perturblab/scgpt-human", scgpt_model)
        args.scgpt_model = str(scgpt_model)

    scfoundation_repo = cache_dir / "scFoundation"
    scfoundation_ckpt_dir = cache_dir / "scFoundation_hf"
    scfoundation_ckpt = scfoundation_ckpt_dir / "models.ckpt"
    need_default_repo = run_all_by_default or (args.scfoundation_ckpt and not args.scfoundation_repo)
    need_default_ckpt = run_all_by_default or (args.scfoundation_repo and not args.scfoundation_ckpt)
    if need_default_repo:
        if not (scfoundation_repo / "model" / "get_embedding.py").exists():
            print("[models] Cloning scFoundation repo...")
            scfoundation_repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call([
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/biomap-research/scFoundation",
                str(scfoundation_repo),
            ])
        args.scfoundation_repo = str(scfoundation_repo)
    if need_default_ckpt:
        if not scfoundation_ckpt.exists():
            print("[models] Downloading scFoundation checkpoint...")
            _download_hf_snapshot("genbio-ai/scFoundation", scfoundation_ckpt_dir)
        args.scfoundation_ckpt = str(scfoundation_ckpt)


def build_geneformer_ensembl_map(adata: ad.AnnData, model_dir: str) -> Optional[Dict[str, str]]:
    dictionary_dir = Path(model_dir).resolve().parent / "geneformer"
    candidates = [
        dictionary_dir / "gene_name_id_dict_gc104M.pkl",
        Path(__file__).resolve().parent
        / DEFAULT_ENV_DIR
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "geneformer"
        / "gene_name_id_dict.pkl",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                name_to_ensembl = pickle.load(f)
            return {
                gene: name_to_ensembl[gene]
                for gene in adata.var_names
                if gene in name_to_ensembl
            }
    return None


def run_bakeoff(args):
    ensure_public_model_assets(args)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = prepare_eval_set(
        args.adata,
        n_cells=args.n_cells,
        cell_type_col=args.cell_type_col,
        time_col=args.time_col,
        qc_done=args.qc_done,
        seed=args.seed,
    )
    adata.write_h5ad(output_dir / "eval_set.h5ad")

    # Optional ensembl mapping for Geneformer
    ensembl_map = None
    if args.ensembl_map and Path(args.ensembl_map).exists():
        ensembl_map = pd.read_csv(args.ensembl_map).set_index("symbol")["ensembl_id"].to_dict()
    elif args.geneformer_model:
        ensembl_map = build_geneformer_ensembl_map(adata, args.geneformer_model)
        if ensembl_map:
            print(f"[Geneformer] Auto-mapped {len(ensembl_map)}/{adata.n_vars} genes to Ensembl IDs")

    extractors: Dict[str, Callable[[ad.AnnData], np.ndarray]] = {}
    if args.geneformer_model:
        extractors["geneformer_v2"] = lambda a: get_geneformer_embeddings(
            a, model_dir=args.geneformer_model, ensembl_map=ensembl_map
        )
    if args.scgpt_model:
        extractors["scgpt"] = lambda a: get_scgpt_embeddings(
            a, model_dir=args.scgpt_model, gene_col=args.gene_col
        )
    if args.scfoundation_repo and args.scfoundation_ckpt:
        extractors["scfoundation"] = lambda a: get_scfoundation_embeddings(
            a, repo_dir=args.scfoundation_repo, model_ckpt=args.scfoundation_ckpt
        )

    if not extractors:
        raise SystemExit("No models specified. Pass at least one --<model>-... flag.")

    results = {}
    for name, extractor in extractors.items():
        print(f"\n{'='*60}\nMODEL: {name}\n{'='*60}")
        emb_path = output_dir / f"{name}_emb.npy"
        if emb_path.exists() and not args.force:
            print(f"  Loading cached embeddings from {emb_path}")
            embeddings = np.load(emb_path)
            if embeddings.ndim != 2 or embeddings.shape[0] != adata.n_obs:
                print(
                    f"  Cached embeddings have shape {embeddings.shape}, but "
                    f"current eval set has {adata.n_obs} cells; recomputing."
                )
                embeddings = extractor(adata.copy())
                np.save(emb_path, embeddings)
        else:
            embeddings = extractor(adata.copy())
            np.save(emb_path, embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != adata.n_obs:
            raise ValueError(
                f"{name} embeddings must have shape (n_cells, n_features) with "
                f"{adata.n_obs} rows after filtering/subsampling; got {embeddings.shape}."
            )

        print(f"  Evaluating {name}...")
        probe = linear_probe_eval(
            embeddings, adata.obs[args.cell_type_col].values, seed=args.seed
        )
        ptime = pseudotime_eval(
            embeddings, adata.obs[args.time_col].values
        ) if args.time_col in adata.obs.columns else {"note": "no time column"}
        sep = cluster_separation_eval(
            embeddings,
            adata.obs[args.cell_type_col].values,
            focal_labels=tuple(args.focal_labels.split(",")),
        )
        results[name] = {
            "linear_probe": probe,
            "pseudotime": ptime,
            "cluster_separation": sep,
            "embedding_dim": int(embeddings.shape[1]),
        }
        print(json.dumps(results[name], indent=2))

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if len(results) > 1:
        make_comparison_plot(results, str(output_dir / "comparison.png"))

    print(f"\nAll results saved to {output_dir}")
    return results

if __name__ == "__main__":
    args = parse_args()
    run_bakeoff(args)
