# scfm-triage

Triage single-cell foundation models on your own labeled AnnData before you
commit to a base model.

`foundation_model_triage.py` downloads public model assets, extracts zero-shot
cell embeddings from Geneformer V2, scGPT, and scFoundation, then evaluates
which base looks strongest on your data.

## What It Tests

The script runs three checks:

1. Linear probe cell type classification accuracy from frozen embeddings.
2. Diffusion pseudotime recovery against a known time or stage column.
3. UMAP cluster separation for a pair of focal labels, such as
   `iPSC,partial_reprog`.

The focal-label test is optional in practice: the default is useful for iPSC
reprogramming data, but any two labels from your cell type column can be used.

## Quick Start

```bash
python foundation_model_triage.py \
  --adata /path/to/your_data.h5ad \
  --qc-done \
  --cell-type-col cell_type \
  --time-col time_point \
  --output triage_results
```

For iPSC reprogramming data:

```bash
python foundation_model_triage.py \
  --adata /path/to/your_data.h5ad \
  --qc-done \
  --cell-type-col cell_type \
  --time-col time_point \
  --focal-labels iPSC,partial_reprog \
  --n-cells 100000 \
  --output triage_results
```

For another dataset, change the focal labels:

```bash
--focal-labels naive_T,effector_T
```

## Inputs

Required:

- `--adata`: path to an `.h5ad` AnnData file.
- One QC choice:
  - `--qc-done`: your AnnData is already QC-filtered.
  - `--run-qc`: run the script's basic QC filters before evaluation.
- `--cell-type-col`: column in `adata.obs` with clear cell type labels.
- `--time-col`: column in `adata.obs` with time, stage, or ordering labels.

Useful optional arguments:

- `--n-cells`: maximum number of cells to evaluate after the selected QC behavior.
- `--focal-labels`: two comma-separated labels for cluster separation.
- `--output`: output directory.
- `--model-cache`: directory for downloaded public model assets.
- `--force`: recompute embeddings even if cached outputs exist.

## What `--n-cells` Means

`--n-cells` is a runtime and memory cap. If your dataset has more cells after
the selected QC behavior, the script stratified-subsamples by `--cell-type-col`.
If your dataset has fewer cells, it uses all available cells.

Suggested values:

- First smoke test: `--n-cells 1000`
- Practical trial: `--n-cells 5000` to `10000`
- Final triage run: `--n-cells 50000` to `100000`

## Outputs

The output directory contains:

- `results.json`: metrics for each model.
- `comparison.png`: summary plot when more than one model is evaluated.
- `geneformer_v2_emb.npy`: cached Geneformer V2 embeddings.
- `scgpt_emb.npy`: cached scGPT embeddings.
- `scfoundation_emb.npy`: cached scFoundation embeddings.

## Automatic Setup

By default, the script creates an isolated environment:

```text
.foundation_model_triage_env
```

It also downloads or reuses model assets under:

```text
public_models
```

The first run can take a while because models and Python packages are installed.
Later runs reuse the environment, model cache, and embeddings unless `--force`
is passed.

To only prepare the environment:

```bash
python foundation_model_triage.py --setup-only
```

## Model Assets

If no model paths are provided, the script uses public defaults:

- Geneformer V2 104M
- scGPT human
- scFoundation

Advanced users can override paths:

```bash
--geneformer-model /path/to/Geneformer-V2-104M
--ensembl-map /path/to/symbol_to_ensembl.csv
--scgpt-model /path/to/scgpt-human
--scfoundation-repo /path/to/scFoundation
--scfoundation-ckpt /path/to/models.ckpt
```

Most users should not need these flags.

## Interpreting Results

Use this as an empirical model selection screen, not as a final biological
claim. A strong base model should preserve known cell type structure, recover
known temporal structure when present, and separate biologically important
states without supervised fine-tuning.
