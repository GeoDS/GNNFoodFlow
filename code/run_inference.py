"""
Generate county-to-county FoodFlow predictions with the multitask LocalizedGCN.

The model was trained on FAF zone flows. County inference uses county feature
projections and a county k-NN graph stored in artifacts/.

Usage:
  python run_inference.py --all-county-crossscale
  python run_inference.py --all-county-crossscale --out predictions_county_crossscale.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(ROOT, "artifacts")
MODEL_PATH = os.path.join(ROOT, "models", "best_mt_localized_gcn.pth")
DEFAULT_OUT = os.path.join(ROOT, "predictions_county_crossscale.csv")
SCTG_CODES = list(range(1, 8))

sys.path.insert(0, ROOT)
from model import MTLocalizedGCN  # noqa: E402


def is_valid_fips(value):
    """Return True for real five-digit county FIPS codes."""
    return isinstance(value, str) and value.isdigit() and len(value) == 5


def compact_county_artifacts(X_pca, sei, county_to_idx, meta_df):
    """Drop invalid county rows and remap graph indices.

    Some county artifacts may contain placeholder rows such as ``00nan`` from
    upstream joins. Filtering here keeps inference outputs aligned to real
    counties without changing the trained FAF model.
    """
    valid_items = [
        (old_idx, fips)
        for fips, old_idx in county_to_idx.items()
        if is_valid_fips(fips) and fips in meta_df.index
    ]
    valid_items.sort()
    old_to_new = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(valid_items)}

    keep = np.array([old_idx for old_idx, _ in valid_items], dtype=np.int64)
    X_pca = X_pca[keep]

    if sei.numel():
        src = sei[0].cpu().numpy()
        dst = sei[1].cpu().numpy()
        mask = np.isin(src, keep) & np.isin(dst, keep)
        remapped = np.array(
            [[old_to_new[int(s)], old_to_new[int(d)]] for s, d in zip(src[mask], dst[mask])],
            dtype=np.int64,
        )
        sei = torch.tensor(remapped.T, dtype=torch.long) if len(remapped) else torch.empty((2, 0), dtype=torch.long)

    return X_pca, sei, {fips: new_idx for new_idx, (_, fips) in enumerate(valid_items)}


def load_model():
    """Load the trained multitask LocalizedGCN checkpoint."""
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    zone_sei = ckpt["model_state_dict"].get("sparse_edge_index")
    model = MTLocalizedGCN(
        node_dim=ckpt["node_dim"],
        edge_dim=ckpt["edge_dim"],
        hidden=ckpt["hidden"] // 2,
        n_tasks=ckpt["n_tasks"],
        dropout=ckpt.get("dropout", 0.2),
        sparse_edge_index=zone_sei,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_county_artifacts():
    """Load county features, graph, metadata, and edge scaler."""
    import pickle

    required = [
        "county_X_pca.npy",
        "county_sei.pt",
        "county_to_idx.json",
        "county_meta.csv",
        "edge_scaler.pkl",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(ARTIFACTS, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing county artifact(s) in {ARTIFACTS}: {', '.join(missing)}"
        )

    with open(os.path.join(ARTIFACTS, "county_to_idx.json")) as f:
        county_to_idx = json.load(f)
    with open(os.path.join(ARTIFACTS, "edge_scaler.pkl"), "rb") as f:
        edge_scaler = pickle.load(f)

    meta_df = (
        pd.read_csv(os.path.join(ARTIFACTS, "county_meta.csv"), dtype={"FIPS": str})
        .assign(FIPS=lambda d: d["FIPS"].str.zfill(5))
        .loc[lambda d: d["FIPS"].map(is_valid_fips)]
        .drop_duplicates(subset=["FIPS"], keep="first")
        .set_index("FIPS")
    )
    X_pca = np.load(os.path.join(ARTIFACTS, "county_X_pca.npy"))
    sei = torch.load(os.path.join(ARTIFACTS, "county_sei.pt"), map_location="cpu")
    X_pca, sei, county_to_idx = compact_county_artifacts(X_pca, sei, county_to_idx, meta_df)

    return {
        "X_pca": X_pca,
        "sei": sei,
        "county_to_idx": county_to_idx,
        "meta_df": meta_df,
        "edge_scaler": edge_scaler,
    }


def county_meta_arrays(artifacts):
    """Return lat/lon/population/port arrays aligned to county feature rows."""
    n = artifacts["X_pca"].shape[0]
    lat = np.full(n, 39.5, dtype=np.float32)
    lon = np.full(n, -98.0, dtype=np.float32)
    pop = np.zeros(n, dtype=np.float32)
    port = np.zeros(n, dtype=np.float32)
    meta = artifacts["meta_df"]

    for fips, idx in artifacts["county_to_idx"].items():
        if fips not in meta.index:
            continue
        row = meta.loc[fips]
        lat[idx] = float(row["lat"])
        lon[idx] = float(row["lon"])
        pop[idx] = float(row["population"]) if pd.notna(row["population"]) else 0.0
        port[idx] = float(bool(row["has_port"]))

    return lat, lon, pop, port


def county_edge_features(origin_idx, dest_idx, lat, lon, pop, port, edge_scaler):
    """Build scaled edge features for one origin county and many destinations."""
    lat1 = np.radians(lat[origin_idx])
    lat2 = np.radians(lat[dest_idx])
    dlon = np.radians(lon[dest_idx] - lon[origin_idx])
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = np.maximum(2 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1)), 1.0)

    gravity = (pop[origin_idx] * pop[dest_idx]) / (dist**2 + 1e-8)
    raw = np.stack(
        [
            dist,
            np.log1p(dist),
            1.0 / (dist + 1.0),
            gravity,
            np.log1p(gravity),
            np.full(len(dest_idx), port[origin_idx], dtype=np.float32),
            port[dest_idx],
        ],
        axis=1,
    ).astype(np.float32)
    return edge_scaler.transform(raw).astype(np.float32)


@torch.no_grad()
def compute_county_embeddings(model, artifacts):
    """Run the trained GCN layers on county features and the county k-NN graph."""
    x = torch.tensor(artifacts["X_pca"], dtype=torch.float)
    sei = artifacts["sei"]
    model.sparse_edge_index = sei
    h = F.leaky_relu(model.bn1(model.conv1(x, sei)), 0.01)
    h = F.dropout(h, p=0.2, training=False)
    h = F.leaky_relu(model.bn2(model.conv2(h, sei)), 0.01)
    return h


@torch.no_grad()
def predict_from_embeddings(model, h, edge_attr, origin_idx, dest_idx):
    """Run the edge encoder and seven hurdle heads for a batch of county OD pairs."""
    edge_t = torch.tensor(edge_attr, dtype=torch.float)
    h_o = h[origin_idx : origin_idx + 1].expand(len(dest_idx), -1)
    h_d = h[torch.tensor(dest_idx, dtype=torch.long)]
    shared = model.edge_mlp(torch.cat([h_o, h_d, model.edge_encoder(edge_t)], dim=1))

    logits = []
    values = []
    for head in model.heads:
        logit, value = head(shared)
        logits.append(logit)
        values.append(value)

    prob = torch.sigmoid(torch.cat(logits, dim=1)).numpy()
    value = torch.cat(values, dim=1).numpy()
    tons = np.where(prob > 0.5, np.expm1(np.clip(value, 0, 20)), 0.0)
    return tons, prob


def write_all_county_crossscale(model, artifacts, out_path):
    """Stream all directed county-pair predictions to CSV."""
    county_items = sorted((idx, fips) for fips, idx in artifacts["county_to_idx"].items())
    lat, lon, pop, port = county_meta_arrays(artifacts)
    h = compute_county_embeddings(model, artifacts)
    meta = artifacts["meta_df"]

    sctg_cols = [f"sctg{k}_tons" for k in SCTG_CODES]
    prob_cols = [f"sctg{k}_prob" for k in SCTG_CODES]
    first_write = True
    n_rows = 0
    start = time.time()

    for origin_number, (origin_idx, origin_fips) in enumerate(county_items, 1):
        dest_items = [(idx, fips) for idx, fips in county_items if idx != origin_idx]
        dest_idx = np.array([idx for idx, _ in dest_items], dtype=np.int64)
        dest_fips = [fips for _, fips in dest_items]

        edge_attr = county_edge_features(
            origin_idx, dest_idx, lat, lon, pop, port, artifacts["edge_scaler"]
        )
        tons, prob = predict_from_embeddings(model, h, edge_attr, origin_idx, dest_idx)

        origin_meta = meta.loc[origin_fips] if origin_fips in meta.index else {}
        dest_meta = meta.reindex(dest_fips)
        rows = {
            "orig_fips": origin_fips,
            "orig_county": origin_meta.get("county_full", origin_fips),
            "orig_state": origin_meta.get("state_abbr", ""),
            "dest_fips": dest_fips,
            "dest_county": dest_meta["county_full"].fillna("").values,
            "dest_state": dest_meta["state_abbr"].fillna("").values,
        }
        for i, col in enumerate(sctg_cols):
            rows[col] = tons[:, i].astype(np.float32)
        for i, col in enumerate(prob_cols):
            rows[col] = prob[:, i].astype(np.float32)
        rows["total_tons"] = tons.sum(axis=1).astype(np.float32)

        pd.DataFrame(rows).to_csv(
            out_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False
        n_rows += len(dest_idx)

        if origin_number % 100 == 0 or origin_number == len(county_items):
            elapsed = (time.time() - start) / 60
            print(
                f"  {origin_number:,}/{len(county_items):,} origins | "
                f"{n_rows:,} rows | {elapsed:.1f} min",
                flush=True,
            )

    return n_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate county cross-scale predictions.")
    parser.add_argument(
        "--all-county-crossscale",
        action="store_true",
        help="Generate all directed county-to-county predictions.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.all_county_crossscale:
        raise SystemExit("Use --all-county-crossscale to generate predictions.")

    print("Loading county artifacts...")
    artifacts = load_county_artifacts()
    print(f"  {len(artifacts['county_to_idx']):,} counties")

    print("Loading model...")
    model, ckpt = load_model()
    r2 = ckpt.get("test_metrics", {}).get("mean_r2", float("nan"))
    print(f"  test mean R2: {r2:.4f}")

    print(f"Writing predictions to {args.out}")
    n_rows = write_all_county_crossscale(model, artifacts, args.out)
    print(f"Saved {n_rows:,} rows to {args.out}")


if __name__ == "__main__":
    main()
