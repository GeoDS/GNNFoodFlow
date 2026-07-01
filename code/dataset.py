"""
dataset.py — Multitask data loading for FoodFlow multitask.

Loads all 7 SCTG datasets, aligns them to a union edge set, and stacks
targets into edge_y of shape (E, 7).

Interface compatibility with v2_parallel:
  - All dataset dict keys are identical
  - Only edge_y changes shape: (E, 1) → (E, 7)
  - x, edge_index, edge_attr, sparse_edge_index are unchanged
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

SCTG_CODES = list(range(1, 8))


# ─────────────────────────────────────────────────────────────
# Local node and sparse graph utilities
# ─────────────────────────────────────────────────────────────

def load_faf_node_features(n_pca: int = 30):
    """Load and PCA-reduce FAF-zone features from this repo's code/data folder."""
    node_path = os.path.join(DATA_DIR, "faf_features_aligned_filtered.csv")
    df = pd.read_csv(node_path)
    df["FAF_Zone"] = df["FAF_Zone"].astype(str).str.zfill(3)

    faf_zones = df["FAF_Zone"].tolist()
    zone_to_idx = {z: i for i, z in enumerate(faf_zones)}
    population = df["population"].values.astype(np.float64)

    # Minimal core package does not bundle port shapefiles. County inference
    # still uses the saved county metadata/artifacts with county port flags.
    port_flags = np.zeros(len(faf_zones), dtype=np.int32)

    feature_cols = [c for c in df.columns if c != "FAF_Zone"]
    x = df[feature_cols].values.astype(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=n_pca, random_state=42)
    x_pca = pca.fit_transform(x_scaled).astype(np.float32)

    print(
        f"[dataset] {len(faf_zones)} FAF zones, {x_pca.shape[1]}-dim PCA "
        f"({pca.explained_variance_ratio_.sum() * 100:.1f}% var)"
    )
    return faf_zones, zone_to_idx, x_pca, scaler, pca, port_flags, population


def _build_sparse_knn(dist_matrix, faf_zones, zone_to_idx, k=10):
    """Build an undirected distance-kNN graph over FAF zones."""
    dist_matrix = dist_matrix.copy()
    dist_matrix["FAF_Zone"] = dist_matrix["FAF_Zone"].astype(str).str.zfill(3)
    dist_matrix = dist_matrix.set_index("FAF_Zone")

    n = len(faf_zones)
    dmat = np.zeros((n, n), dtype=np.float32)
    for z, i in zone_to_idx.items():
        if z in dist_matrix.index:
            for z2, j in zone_to_idx.items():
                if z2 in dist_matrix.columns:
                    try:
                        dmat[i, j] = float(dist_matrix.loc[z, z2])
                    except Exception:
                        pass

    src, dst = [], []
    for i in range(n):
        d = dmat[i].copy()
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            src += [i, int(j)]
            dst += [int(j), i]
    return torch.unique(torch.tensor([src, dst], dtype=torch.long), dim=1)


# ─────────────────────────────────────────────────────────────
# Build union edge set across all 7 SCTGs
# ─────────────────────────────────────────────────────────────

def _load_raw_flows(codes: list, zone_to_idx: dict) -> dict:
    """
    Load raw tons_2017 for each SCTG code.
    Returns dict: code → pd.DataFrame with columns [dms_orig, dms_dest, tons_2017]
    """
    raw = {}
    for code in codes:
        path = os.path.join(DATA_DIR, f"FAF5_SCTG{code}.csv")
        df = pd.read_csv(path)[["dms_orig", "dms_dest", "tons_2017"]]
        df = df.groupby(["dms_orig", "dms_dest"])["tons_2017"].sum().reset_index()
        df["dms_orig"] = df["dms_orig"].astype(str).str.zfill(3)
        df["dms_dest"] = df["dms_dest"].astype(str).str.zfill(3)
        # Keep only pairs where both zones are in our node set
        valid = df["dms_orig"].isin(zone_to_idx) & df["dms_dest"].isin(zone_to_idx)
        raw[code] = df[valid].copy()
        print(f"[dataset] SCTG {code}: {len(df[valid])} OD pairs with flow")
    return raw


def _build_union_edge_set(raw_flows: dict, zone_to_idx: dict) -> list:
    """
    Build the full union of all OD pairs across all SCTG codes.
    Self-loops (orig == dest) are included only if present in raw data.
    Returns list of (orig_str, dest_str) tuples.
    """
    all_pairs = set()
    for df in raw_flows.values():
        for o, d in zip(df["dms_orig"], df["dms_dest"]):
            all_pairs.add((o, d))
    pairs = sorted(all_pairs)
    print(f"[dataset] Union edge set: {len(pairs)} OD pairs")
    return pairs


def _build_all_directed_edge_set(zone_to_idx: dict, include_self: bool = False) -> list:
    """Build all directed FAF OD pairs for strict transductive evaluation."""
    zones = sorted(zone_to_idx)
    pairs = [
        (o, d)
        for o in zones
        for d in zones
        if include_self or o != d
    ]
    print(f"[dataset] All-pairs edge set: {len(pairs)} OD pairs")
    return pairs


# ─────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────

def load_multitask_dataset(
    codes:   list = None,
    n_pca:   int  = 30,
    knn_k:   int  = 10,
    edge_universe: str = "all_pairs",
) -> dict:
    """
    Load all SCTG datasets aligned to a union edge set.

    Returns a single dataset dict compatible with v2_parallel format:
    {
        "x":                 Tensor (N, n_pca)       — node features
        "edge_index":        Tensor (2, E)            — directed edges
        "edge_attr":         Tensor (E, 7)            — raw edge features; split will scale train-only
        "edge_y":            Tensor (E, 7)            — log1p(tons) per SCTG
        "sparse_edge_index": Tensor (2, E_sparse)     — k-NN graph for GCN
        "faf_zones":         list[str]
        "zone_to_idx":       dict[str, int]
    }
    """
    if codes is None:
        codes = SCTG_CODES

    # ── 1. Shared node features ────────────────────────────────────────────────
    print("[dataset] Loading node features...")
    faf_zones, zone_to_idx, X_pca, node_scaler, node_pca, port_flags, population = \
        load_faf_node_features(n_pca=n_pca)

    dist_matrix = pd.read_csv(os.path.join(DATA_DIR, "FAF_distance_matrix.csv"))

    # ── 2. Raw flows per SCTG ─────────────────────────────────────────────────
    print("[dataset] Loading raw SCTG flows...")
    raw_flows = _load_raw_flows(codes, zone_to_idx)

    # ── 3. Candidate OD edge set ──────────────────────────────────────────────
    if edge_universe == "all_pairs":
        union_pairs = _build_all_directed_edge_set(zone_to_idx, include_self=False)
    elif edge_universe == "union_support":
        union_pairs = _build_union_edge_set(raw_flows, zone_to_idx)
    else:
        raise ValueError("edge_universe must be 'all_pairs' or 'union_support'")
    E = len(union_pairs)

    # Build index arrays for union edges
    o_strs = [p[0] for p in union_pairs]
    d_strs = [p[1] for p in union_pairs]
    o_idx  = np.array([zone_to_idx[o] for o in o_strs], dtype=np.int64)
    d_idx  = np.array([zone_to_idx[d] for d in d_strs], dtype=np.int64)

    # ── 4. Edge features (geography only, same for all tasks) ─────────────────
    print("[dataset] Computing edge features...")
    union_df = pd.DataFrame({
        "dms_orig":  o_strs,
        "dms_dest":  d_strs,
        "tons_2017": np.zeros(E, dtype=np.float32),  # placeholder; not used here
    })
    # Build edge_index directly to guarantee alignment with the candidate edge set.
    # Edge attributes are kept raw here and scaled only after the split.
    edge_index = torch.tensor(np.stack([o_idx, d_idx]), dtype=torch.long)

    # Compute raw edge features with guaranteed alignment.
    dist_matrix_copy = dist_matrix.copy()
    dist_matrix_copy["FAF_Zone"] = dist_matrix_copy["FAF_Zone"].astype(str).str.zfill(3)
    dist_matrix_copy = dist_matrix_copy.set_index("FAF_Zone")

    N = len(faf_zones)
    D = np.zeros((N, N), dtype=np.float32)
    for z, i in zone_to_idx.items():
        if z in dist_matrix_copy.columns:
            for z2, j in zone_to_idx.items():
                if z2 in dist_matrix_copy.columns:
                    try:
                        D[i, j] = float(dist_matrix_copy.loc[z, z2])
                    except Exception:
                        pass

    dist        = D[o_idx, d_idx]
    pop_o       = population[o_idx]
    pop_d       = population[d_idx]
    port_o      = port_flags[o_idx].astype(float)
    port_d      = port_flags[d_idx].astype(float)
    log_dist    = np.log1p(dist)
    inv_dist    = 1.0 / (dist + 1.0)
    gravity     = (pop_o * pop_d) / (dist ** 2 + 1e-8)
    log_gravity = np.log1p(gravity)

    raw_feat = np.stack(
        [dist, log_dist, inv_dist, gravity, log_gravity, port_o, port_d], axis=1
    ).astype(np.float32)

    edge_attr = torch.tensor(raw_feat, dtype=torch.float)

    # ── 5. Stack targets (E, 7) ───────────────────────────────────────────────
    print("[dataset] Building multitask targets (E, 7)...")

    # Build lookup: (orig_str, dest_str) → tons for each SCTG
    pair_to_union_idx = {(o, d): i for i, (o, d) in enumerate(union_pairs)}

    y_matrix = np.zeros((E, len(codes)), dtype=np.float32)

    for k, code in enumerate(codes):
        df = raw_flows[code]
        for _, row in df.iterrows():
            key = (row["dms_orig"], row["dms_dest"])
            if key in pair_to_union_idx:
                idx = pair_to_union_idx[key]
                y_matrix[idx, k] = float(row["tons_2017"])

    edge_y = torch.tensor(np.log1p(y_matrix), dtype=torch.float)  # (E, 7)

    n_pos_any = (edge_y.sum(dim=1) > 0).sum().item()
    print(f"[dataset] edge_y shape: {tuple(edge_y.shape)}, "
          f"{n_pos_any}/{E} edges have flow in at least 1 SCTG "
          f"({100*n_pos_any/E:.1f}%)")
    for k, code in enumerate(codes):
        n_pos_k = (edge_y[:, k] > 0).sum().item()
        print(f"          SCTG {code}: {n_pos_k} positive ({100*n_pos_k/E:.1f}%)")

    # ── 6. Sparse k-NN graph ──────────────────────────────────────────────────
    print("[dataset] Building sparse k-NN graph...")
    sparse_ei = _build_sparse_knn(dist_matrix, faf_zones, zone_to_idx, k=knn_k)

    x = torch.tensor(X_pca, dtype=torch.float)

    return {
        "x":                 x,
        "edge_index":        edge_index,
        "edge_attr":         edge_attr,
        "edge_y":            edge_y,
        "sparse_edge_index": sparse_ei,
        "edge_attr_raw":     edge_attr,
        "faf_zones":         faf_zones,
        "zone_to_idx":       zone_to_idx,
        "codes":             codes,
        "edge_universe":     edge_universe,
    }


# ─────────────────────────────────────────────────────────────
# Train/test split (multitask version)
# ─────────────────────────────────────────────────────────────

def stratified_split_multitask(
    dataset:     dict,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.0,
    seed:        int   = 42,
) -> tuple:
    """
    Stratified split for multitask dataset.

    Stratification criterion: whether ANY of the 7 SCTG tasks has a
    positive flow on that edge (same spirit as v2_parallel's split).

    Returns (train_data, test_data) when val_ratio is 0, otherwise
    (train_data, val_data, test_data). Edge features are scaled using train
    edges only, then applied to validation/test edges.
    """
    edge_index = dataset["edge_index"]
    edge_attr_raw = dataset.get("edge_attr_raw", dataset["edge_attr"])
    edge_y     = dataset["edge_y"]
    x          = dataset["x"]
    sparse_ei  = dataset["sparse_edge_index"]

    rng      = torch.Generator().manual_seed(seed)
    any_pos  = edge_y.sum(dim=1) > 0          # (E,) bool

    zero_idx    = (~any_pos).nonzero(as_tuple=True)[0]
    nonzero_idx = any_pos.nonzero(as_tuple=True)[0]

    def _split_train_test(idx):
        perm = idx[torch.randperm(len(idx), generator=rng)]
        n    = int(len(perm) * train_ratio)
        return perm[:n], perm[n:]

    z_tr, z_te   = _split_train_test(zero_idx)
    nz_tr, nz_te = _split_train_test(nonzero_idx)

    tr_idx = torch.cat([z_tr, nz_tr])
    heldout_idx = torch.cat([z_te, nz_te])
    tr_idx = tr_idx[torch.randperm(len(tr_idx), generator=rng)]
    heldout_idx = heldout_idx[torch.randperm(len(heldout_idx), generator=rng)]

    val_idx = None
    te_idx = heldout_idx
    if val_ratio > 0:
        heldout_fraction = max(1e-12, 1.0 - train_ratio)
        val_fraction = min(max(val_ratio / heldout_fraction, 0.0), 0.95)
        n_val = int(len(heldout_idx) * val_fraction)
        val_idx = heldout_idx[:n_val]
        te_idx = heldout_idx[n_val:]

    scaler = StandardScaler()
    train_edge_attr = scaler.fit_transform(edge_attr_raw[tr_idx].numpy()).astype(np.float32)

    def _scaled_attr(idx):
        if torch.equal(idx, tr_idx):
            arr = train_edge_attr
        else:
            arr = scaler.transform(edge_attr_raw[idx].numpy()).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float)

    def _sub(idx):
        return {
            "x":                 x,
            "edge_index":        edge_index[:, idx],
            "edge_attr":         _scaled_attr(idx),
            "edge_y":            edge_y[idx],
            "sparse_edge_index": sparse_ei,
            "edge_indices":      idx,
            "edge_scaler":       scaler,
        }

    train_data = _sub(tr_idx)
    val_data   = _sub(val_idx) if val_idx is not None else None
    test_data  = _sub(te_idx)

    msg = (
        f"[dataset] Train: {len(tr_idx)} edges "
        f"({(edge_y[tr_idx].sum(dim=1) > 0).sum().item()} with any flow)"
    )
    if val_idx is not None:
        msg += (
            f" | Val: {len(val_idx)} edges "
            f"({(edge_y[val_idx].sum(dim=1) > 0).sum().item()} with any flow)"
        )
    msg += (
        f" | Test: {len(te_idx)} edges "
        f"({(edge_y[te_idx].sum(dim=1) > 0).sum().item()} with any flow)"
    )
    print(msg)

    if val_data is not None:
        return train_data, val_data, test_data
    return train_data, test_data
