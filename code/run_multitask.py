"""
run_multitask.py — Entry point for MT-LocalizedGCN (Multi-Type Food Flow) training.

Usage:
  python run_multitask.py
  python run_multitask.py --epochs 2000 --hidden 256
"""

import sys
import os
import argparse
import pickle
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset import load_multitask_dataset, stratified_split_multitask
from model   import MTEdgeMLP, MTLocalizedGCN
from train   import (
    training_loop,
    compute_pos_weights,
    eval_mt_edge_mlp,
    eval_mt_localized_gcn,
    SCTG_NAMES,
)


def parse_args():
    p = argparse.ArgumentParser(description="FoodFlow multitask training")
    p.add_argument("--model",    default="localized_gcn",
                   choices=["edge_mlp", "localized_gcn"],
                   help="Model architecture (default: localized_gcn)")
    p.add_argument("--epochs",   type=int,   default=1000)
    p.add_argument("--hidden",   type=int,   default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--dropout",  type=float, default=0.2)
    p.add_argument("--n_pca",    type=int,   default=30)
    p.add_argument("--knn_k",    type=int,   default=5)
    p.add_argument("--edge_universe", default="all_pairs",
                   choices=["all_pairs", "union_support"],
                   help="Candidate FAF OD set. all_pairs is the paper/reproducible default.")
    p.add_argument("--train_ratio", type=float, default=0.70,
                   help="Train share of all edges. Default matches paper all-pairs protocol.")
    p.add_argument("--val_ratio", type=float, default=0.15,
                   help="Validation share of all edges, carved from the non-train split.")
    p.add_argument("--selection_metric", default="mean_cpc",
                   choices=["mean_cpc", "mean_r2", "mean_auc", "mean_pos_r2"],
                   help="Validation metric used to select the saved checkpoint.")
    p.add_argument("--log_every",type=int,   default=100)
    p.add_argument("--no_pos_weight", action="store_true",
                   help="Disable positive class weighting")
    p.add_argument("--no_save", action="store_true",
                   help="Run training/evaluation without writing model or scaler artifacts.")
    return p.parse_args()


def print_final_metrics(metrics, label="Test"):
    print(f"\n{'─'*60}")
    print(f"{label} metrics (multitask):")
    print(f"  Mean R²:     {metrics['mean_r2']:.4f}")
    print(f"  Mean AUC:    {metrics['mean_auc']:.4f}")
    print(f"  Mean CPC:    {metrics['mean_cpc']:.4f}")
    print(f"  Mean pos-R²: {metrics['mean_pos_r2']:.4f}")
    print(f"\n  Per-task breakdown:")
    for code in sorted(metrics["per_task"]):
        m    = metrics["per_task"][code]
        name = SCTG_NAMES.get(code, f"SCTG{code}")
        print(
            f"  SCTG {code} {name:<22} "
            f"R²={m['r2']:+.4f}  "
            f"AUC={m['auc']:.4f}  "
            f"CPC={m['cpc']:.4f}  "
            f"pos-R²={m['pos_r2']:+.4f}"
        )
    print(f"{'─'*60}\n")


def main():
    args = parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("MT-LocalizedGCN — Multi-Type Food Flow Training")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Epochs:  {args.epochs}  Hidden: {args.hidden}  LR: {args.lr}")
    print()

    dataset = load_multitask_dataset(
        n_pca=args.n_pca,
        knn_k=args.knn_k,
        edge_universe=args.edge_universe,
    )
    split_data = stratified_split_multitask(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    if args.val_ratio > 0:
        train_data, val_data, test_data = split_data
    else:
        train_data, test_data = split_data
        val_data = None

    node_dim = dataset["x"].shape[1]     # 30
    edge_dim = dataset["edge_attr"].shape[1]  # 7
    n_tasks  = dataset["edge_y"].shape[1]     # 7

    print(f"\nnode_dim={node_dim}, edge_dim={edge_dim}, n_tasks={n_tasks}")
    split_msg = f"Train edges: {train_data['edge_index'].shape[1]} | "
    if val_data is not None:
        split_msg += f"Val edges: {val_data['edge_index'].shape[1]} | "
    split_msg += f"Test edges:  {test_data['edge_index'].shape[1]}"
    print(split_msg + "\n")

    # ── 2. Build model ────────────────────────────────────────────────────────
    if args.model == "edge_mlp":
        model = MTEdgeMLP(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden=args.hidden,
            n_tasks=n_tasks,
            dropout=args.dropout,
        )
    else:
        model = MTLocalizedGCN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden=args.hidden // 2,   # LocalizedGCN uses hidden differently
            n_tasks=n_tasks,
            dropout=args.dropout,
            sparse_edge_index=dataset["sparse_edge_index"],
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}  |  Parameters: {total_params:,}\n")

    # ── 3. Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── 4. Positive weights ───────────────────────────────────────────────────
    pos_weights = None
    if not args.no_pos_weight:
        pos_weights = compute_pos_weights(train_data["edge_y"])
        pw_vals = [f"{pw.item():.1f}" for pw in pos_weights]
        print(f"Pos weights: {pw_vals}\n")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    history = training_loop(
        model       = model,
        train_data  = train_data,
        test_data   = test_data,
        optimizer   = optimizer,
        scheduler   = scheduler,
        epochs      = args.epochs,
        log_every   = args.log_every,
        model_type  = args.model,
        pos_weights = pos_weights,
        val_data    = val_data,
        selection_metric = args.selection_metric,
    )

    # ── 6. Final metrics ──────────────────────────────────────────────────────
    eval_fn = (eval_mt_edge_mlp if args.model == "edge_mlp"
               else eval_mt_localized_gcn)

    with torch.no_grad():
        train_metrics = eval_fn(model, train_data)
        val_metrics   = eval_fn(model, val_data) if val_data is not None else None
        test_metrics  = eval_fn(model, test_data)

    print_final_metrics(train_metrics, label="Train")
    if val_metrics is not None:
        print_final_metrics(val_metrics, label="Val")
    print_final_metrics(test_metrics,  label="Test")

    if args.no_save:
        print("Skipping model/scaler save because --no_save was set.")
        return

    artifacts_dir = os.path.join(ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "edge_scaler.pkl"), "wb") as f:
        pickle.dump(train_data["edge_scaler"], f)

    # ── 7. Save model ─────────────────────────────────────────────────────────
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, f"best_mt_{args.model}.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_type":       args.model,
        "node_dim":         node_dim,
        "edge_dim":         edge_dim,
        "hidden":           args.hidden,
        "n_tasks":          n_tasks,
        "dropout":          args.dropout,
        "edge_universe":    args.edge_universe,
        "selection_metric": args.selection_metric,
        "val_metrics":      val_metrics,
        "test_metrics":     test_metrics,
        "history":          history,
        "codes":            dataset["codes"],
        "faf_zones":        dataset["faf_zones"],
    }, save_path)
    print(f"Model saved → {save_path}")


if __name__ == "__main__":
    main()
