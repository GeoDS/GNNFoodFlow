# Reproduce Core Results

This folder is organized as a minimal core package. It contains the code and
small artifacts needed to train the multitask FAF model and run county
cross-scale inference.

## Environment

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, use the wheel instructions appropriate for your local
PyTorch and Python versions if the plain install does not resolve cleanly.

## Smoke Test

```bash
python code/run_multitask.py \
  --epochs 1 \
  --hidden 32 \
  --log_every 1 \
  --no_pos_weight \
  --no_save
```

## Train

```bash
python code/run_multitask.py
```

The default configuration uses:

- all directed non-self FAF OD pairs;
- train-only edge feature scaling;
- train, validation, and test split;
- validation `mean_cpc` checkpoint selection;
- distance-kNN localized GCN (MT-LocalizedGCN) with `k=5`;
- seven SCTG task-specific hurdle heads.

## County Inference

```bash
python code/run_inference.py \
  --all-county-crossscale \
  --out predictions_county_crossscale.csv
```

The output file is intentionally ignored by git because it is large.

## Included Result Summaries

Small CSV summaries live in `code/results/`:

- `allpairs_transductive_ablation.csv`
- `allpairs_model_baselines.csv`
- `allpairs_transductive_ablation_per_task.csv`
- `knn_ablation_results.csv`

These are lightweight sanity-check summaries, not a full paper artifact bundle.

