"""
train.py — Training and evaluation utilities for FoodFlow multitask.

Loss:   sum of 7 per-task hurdle losses (each with independent alpha weighting)
Metrics: per-task + macro-averaged r2, auc, cpc, pos_r2
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats as _scipy_stats

SCTG_NAMES = {
    1: "Live Animals/Fish",
    2: "Cereal Grains",
    3: "Other Ag Products",
    4: "Animal Feed & Seeds",
    5: "Meat/Seafood",
    6: "Milled Grain Products",
    7: "Other Foodstuffs",
}


# ─────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────

def _hurdle_loss_one(logit, value, y, pos_weight=None):
    """
    Single-task hurdle loss (identical to hurdle_loss).

    logit : (B, 1)
    value : (B, 1)
    y     : (B, 1)  log1p(tons), 0 = no flow
    """
    y_flat = y.squeeze()
    exists = (y_flat > 0).float().unsqueeze(1)

    cls_loss = F.binary_cross_entropy_with_logits(logit, exists, pos_weight=pos_weight)

    mask = exists.bool().squeeze()
    if mask.sum() > 0:
        reg_loss = F.mse_loss(value[mask], y[mask])
    else:
        reg_loss = torch.tensor(0.0, device=logit.device)

    alpha = cls_loss / (cls_loss + reg_loss + 1e-8)
    loss  = alpha * cls_loss + (1 - alpha) * reg_loss
    return loss, cls_loss.item(), reg_loss.item()


def multitask_hurdle_loss(logit, value, y, pos_weights=None, log_sigma2=None):
    """
    Sum of per-task hurdle losses with optional Kendall uncertainty weighting.

    logit      : (E, n_tasks)
    value      : (E, n_tasks)
    y          : (E, n_tasks)
    pos_weights: list of n_tasks tensors (or None)
    log_sigma2 : (n_tasks,) learnable tensor of log(σ²) per task (or None)
                 Kendall et al. (2018) weighting:
                   L_k_weighted = 0.5 * exp(-s_k) * L_k + 0.5 * s_k
                 σ large → task downweighted automatically.

    Returns
    -------
    total_loss   : scalar tensor
    per_task_cls : list[float]
    per_task_reg : list[float]
    """
    n_tasks = logit.shape[1]
    total = torch.tensor(0.0, device=logit.device)
    per_cls = []
    per_reg = []

    for k in range(n_tasks):
        pw = pos_weights[k] if pos_weights is not None else None
        loss_k, cls_k, reg_k = _hurdle_loss_one(
            logit[:, k:k+1], value[:, k:k+1], y[:, k:k+1], pos_weight=pw
        )

        if log_sigma2 is not None:
            s_k    = log_sigma2[k]
            loss_k = 0.5 * torch.exp(-s_k) * loss_k + 0.5 * s_k

        total   = total + loss_k
        per_cls.append(cls_k)
        per_reg.append(reg_k)

    return total, per_cls, per_reg


# ─────────────────────────────────────────────────────────────
# Train steps
# ─────────────────────────────────────────────────────────────

def train_mt_edge_mlp(model, data, optimizer, pos_weights=None):
    """One gradient step for MTEdgeMLP."""
    model.train()
    optimizer.zero_grad()
    logit, value = model(data["x"], data["edge_index"], data["edge_attr"])
    log_sigma2 = getattr(model, "log_sigma2", None)
    loss, _, _ = multitask_hurdle_loss(logit, value, data["edge_y"], pos_weights, log_sigma2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()


def train_mt_localized_gcn(model, data, optimizer, pos_weights=None):
    """One gradient step for MTLocalizedGCN."""
    model.train()
    optimizer.zero_grad()
    logit, value = model(data["x"], data["edge_index"], data["edge_attr"])
    log_sigma2 = getattr(model, "log_sigma2", None)
    loss, _, _ = multitask_hurdle_loss(logit, value, data["edge_y"], pos_weights, log_sigma2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────────────────────
# Eval steps
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_mt_edge_mlp(model, data):
    model.eval()
    logit, value = model(data["x"], data["edge_index"], data["edge_attr"])
    return _compute_multitask_metrics(logit, value, data["edge_y"])


@torch.no_grad()
def eval_mt_localized_gcn(model, data):
    model.eval()
    logit, value = model(data["x"], data["edge_index"], data["edge_attr"])
    return _compute_multitask_metrics(logit, value, data["edge_y"])


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def _compute_metrics_one(logit_k, value_k, y_k):
    """
    Metrics for a single task — identical logic to _compute_metrics.

    logit_k, value_k, y_k : 1-D numpy arrays of length E
    """
    exists_gt   = (y_k > 0).astype(float)
    exists_pred = (logit_k > 0).astype(float)

    accuracy = (exists_pred == exists_gt).mean()
    try:
        auc = roc_auc_score(exists_gt, 1 / (1 + np.exp(-logit_k)))  # sigmoid
    except Exception:
        auc = float("nan")

    pred_tons = np.where(exists_pred > 0, value_k, 0.0)
    mse  = np.mean((pred_tons - y_k) ** 2)
    rmse = np.sqrt(mse)
    var  = np.var(y_k)
    r2   = 1 - mse / (var + 1e-12)

    pos_mask = y_k > 0
    if pos_mask.sum() > 1:
        pos_r2 = 1 - np.mean((value_k[pos_mask] - y_k[pos_mask]) ** 2) / \
                     (np.var(y_k[pos_mask]) + 1e-12)
    else:
        pos_r2 = float("nan")

    pred_exp = np.expm1(np.clip(pred_tons, 0, 30))
    y_exp    = np.expm1(y_k)
    total    = pred_exp.sum() + y_exp.sum()
    cpc      = 2 * np.minimum(pred_exp, y_exp).sum() / (total + 1e-12)

    return {
        "mse":      float(mse),
        "rmse":     float(rmse),
        "r2":       float(r2),
        "pos_r2":   float(pos_r2),
        "accuracy": float(accuracy),
        "auc":      float(auc),
        "cpc":      float(cpc),
    }


def _compute_multitask_metrics(logit, value, y):
    """
    Per-task and macro-averaged metrics.

    logit, value, y : Tensors (E, n_tasks)

    Returns
    -------
    {
        "per_task": { 1: {...}, 2: {...}, ..., 7: {...} },
        "mean_r2":   float,
        "mean_auc":  float,
        "mean_cpc":  float,
        "mean_pos_r2": float,
    }
    """
    logit_np = logit.cpu().numpy()   # (E, n_tasks)
    value_np = value.cpu().numpy()
    y_np     = y.cpu().numpy()

    n_tasks    = logit_np.shape[1]
    per_task   = {}
    r2s, aucs, cpcs, pos_r2s = [], [], [], []

    for k in range(n_tasks):
        code    = k + 1
        metrics = _compute_metrics_one(logit_np[:, k], value_np[:, k], y_np[:, k])
        per_task[code] = metrics
        r2s.append(metrics["r2"])
        aucs.append(metrics["auc"])
        cpcs.append(metrics["cpc"])
        pos_r2s.append(metrics["pos_r2"])

    def _nanmean(lst):
        vals = [v for v in lst if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "per_task":    per_task,
        "mean_r2":     _nanmean(r2s),
        "mean_auc":    _nanmean(aucs),
        "mean_cpc":    _nanmean(cpcs),
        "mean_pos_r2": _nanmean(pos_r2s),
    }


# ─────────────────────────────────────────────────────────────
# Regional aggregation metrics
# ─────────────────────────────────────────────────────────────

def compute_aggregated_metrics(
    pred_log: np.ndarray,
    obs_log:  np.ndarray,
    orig_zones: list,
    dest_zones: list,
) -> dict:
    """
    Aggregate edge-level log1p(tons) predictions to state level and compute metrics.

    FAF zone codes are 3-digit strings (e.g. "011"); the first 2 characters
    identify the state (FIPS code, zero-padded), so "011" → state "01".

    Parameters
    ----------
    pred_log   : (E,) predicted log1p(tons) per edge
    obs_log    : (E,) observed  log1p(tons) per edge
    orig_zones : (E,) origin FAF zone strings
    dest_zones : (E,) destination FAF zone strings

    Returns
    -------
    dict with: n_pairs, cpc, pearson_r, spearman_r, rmse_log, r2_log, vol_error_pct,
               pred_total_tons, obs_total_tons
    """
    pred_tons = np.expm1(np.clip(pred_log, 0, 30))
    obs_tons  = np.expm1(obs_log)

    orig_states = [z[:2] for z in orig_zones]
    dest_states = [z[:2] for z in dest_zones]

    agg_pred: dict = {}
    agg_obs:  dict = {}
    for i in range(len(pred_tons)):
        key = (orig_states[i], dest_states[i])
        agg_pred[key] = agg_pred.get(key, 0.0) + float(pred_tons[i])
        agg_obs[key]  = agg_obs.get(key,  0.0) + float(obs_tons[i])

    all_keys   = sorted(set(agg_pred) | set(agg_obs))
    pred_arr   = np.array([agg_pred.get(k, 0.0) for k in all_keys])
    obs_arr    = np.array([agg_obs.get(k,  0.0) for k in all_keys])

    total = pred_arr.sum() + obs_arr.sum()
    cpc   = float(2 * np.minimum(pred_arr, obs_arr).sum() / (total + 1e-12))

    pred_log_agg = np.log1p(pred_arr)
    obs_log_agg  = np.log1p(obs_arr)

    if len(pred_log_agg) > 1:
        pearson_r  = float(np.corrcoef(pred_log_agg, obs_log_agg)[0, 1])
        spearman_r = float(_scipy_stats.spearmanr(pred_log_agg, obs_log_agg).statistic)
    else:
        pearson_r = spearman_r = float("nan")

    mse  = float(np.mean((pred_log_agg - obs_log_agg) ** 2))
    rmse = float(np.sqrt(mse))
    var  = float(np.var(obs_log_agg))
    r2   = float(1 - mse / (var + 1e-12))

    vol_error_pct = float(
        100 * (pred_arr.sum() - obs_arr.sum()) / (obs_arr.sum() + 1e-12)
    )

    return {
        "n_pairs":        len(all_keys),
        "cpc":            cpc,
        "pearson_r":      pearson_r,
        "spearman_r":     spearman_r,
        "rmse_log":       rmse,
        "r2_log":         r2,
        "vol_error_pct":  vol_error_pct,
        "pred_total_tons": float(pred_arr.sum()),
        "obs_total_tons":  float(obs_arr.sum()),
    }


def compute_port_metrics(
    logit:        np.ndarray,
    value:        np.ndarray,
    y:            np.ndarray,
    orig_zones:   list,
    dest_zones:   list,
    port_zone_set: set,
) -> dict:
    """
    Compute edge-level metrics stratified by port zone involvement.

    Returns a dict with keys:
      "all"       — all edges
      "port_orig" — edges where origin  is a port zone
      "port_dest" — edges where destination is a port zone
      "port_both" — edges where both origin and destination are port zones
      "port_any"  — edges where at least one endpoint is a port zone

    Each value is the result of _compute_metrics_one (or None if <2 edges).
    """
    port_o = np.array([z in port_zone_set for z in orig_zones])
    port_d = np.array([z in port_zone_set for z in dest_zones])

    masks = {
        "all":       np.ones(len(y), dtype=bool),
        "port_orig": port_o,
        "port_dest": port_d,
        "port_both": port_o & port_d,
        "port_any":  port_o | port_d,
    }

    result = {}
    for key, mask in masks.items():
        if mask.sum() >= 2:
            result[key] = _compute_metrics_one(logit[mask], value[mask], y[mask])
            result[key]["n_edges"] = int(mask.sum())
        else:
            result[key] = None
    return result


# ─────────────────────────────────────────────────────────────
# Positive-class weights (for class imbalance)
# ─────────────────────────────────────────────────────────────

def compute_pos_weights(edge_y: torch.Tensor) -> list:
    """
    Compute pos_weight for BCEWithLogitsLoss per task.

    pos_weight is intended for rare positive classes (value >= 1).
    When positives are the majority, we clamp to 1.0 (no reweighting)
    to avoid down-weighting the majority class and inverting gradients.
    """
    weights = []
    for k in range(edge_y.shape[1]):
        y_k   = edge_y[:, k]
        n_pos = (y_k > 0).sum().item()
        n_neg = (y_k == 0).sum().item()
        pw    = max(n_neg / max(n_pos, 1), 1.0)   # clamp: never < 1.0
        weights.append(torch.tensor([pw], dtype=torch.float))
    return weights


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def training_loop(
    model,
    train_data:  dict,
    test_data:   dict,
    optimizer,
    scheduler=None,
    epochs:      int  = 1000,
    log_every:   int  = 100,
    model_type:  str  = "edge_mlp",    # "edge_mlp" | "localized_gcn"
    pos_weights=None,
    val_data:    dict = None,
    selection_metric: str = "mean_cpc",
):
    """
    Run the full multitask training loop.

    Returns
    -------
    history : list of dicts — one entry per logged epoch
    """
    if model_type == "edge_mlp":
        train_fn = train_mt_edge_mlp
        eval_fn  = eval_mt_edge_mlp
    else:
        train_fn = train_mt_localized_gcn
        eval_fn  = eval_mt_localized_gcn

    best_score = -float("inf")
    best_loss  = float("inf")
    best_state = None
    best_epoch = None
    history    = []

    for epoch in range(1, epochs + 1):
        loss = train_fn(model, train_data, optimizer, pos_weights)

        if scheduler is not None:
            scheduler.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            tr_m = eval_fn(model, train_data)
            va_m = eval_fn(model, val_data) if val_data is not None else None
            te_m = eval_fn(model, test_data) if val_data is None else None

            entry = {
                "epoch":          epoch,
                "train_loss":     loss,
                "train_mean_r2":  tr_m["mean_r2"],
            }
            if va_m is not None:
                entry.update({
                    "val_mean_r2": va_m["mean_r2"],
                    "val_mean_auc": va_m["mean_auc"],
                    "val_mean_cpc": va_m["mean_cpc"],
                    "val_mean_pos_r2": va_m["mean_pos_r2"],
                    "val_per_task": va_m["per_task"],
                })
            else:
                entry.update({
                    "test_mean_r2": te_m["mean_r2"],
                    "test_mean_auc": te_m["mean_auc"],
                    "test_mean_cpc": te_m["mean_cpc"],
                    "test_mean_pos_r2": te_m["mean_pos_r2"],
                    "test_per_task": te_m["per_task"],
                })
            history.append(entry)

            # Per-task R² summary line
            report_m = va_m if va_m is not None else te_m
            report_label = "val" if va_m is not None else "test"
            per_r2 = " ".join(
                f"S{k}={report_m['per_task'][k]['r2']:.3f}"
                for k in sorted(report_m["per_task"])
            )
            print(
                f"Epoch {epoch:4d} | loss={loss:.4f} | "
                f"{report_label}R²={report_m['mean_r2']:.4f} | "
                f"{report_label}AUC={report_m['mean_auc']:.4f} | "
                f"{report_label}CPC={report_m['mean_cpc']:.4f}"
            )
            print(f"         per-task R²: {per_r2}")

            # Print learned σ values if uncertainty weighting is active
            log_sigma2 = getattr(model, "log_sigma2", None)
            if log_sigma2 is not None:
                sigmas = torch.exp(0.5 * log_sigma2).detach()
                sigma_str = " ".join(f"S{i+1}={sigmas[i]:.3f}" for i in range(len(sigmas)))
                print(f"         σ (uncertainty): {sigma_str}")

        if val_data is not None:
            with torch.no_grad():
                score_metrics = eval_fn(model, val_data)
            score = score_metrics.get(selection_metric, float("nan"))
            is_better = score > best_score
        else:
            score = -loss
            is_better = loss < best_loss

        if is_better:
            best_score = score
            best_loss  = loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        if val_data is not None:
            print(f"\nRestored best model (epoch={best_epoch}, val_{selection_metric}={best_score:.6f})")
        else:
            print(f"\nRestored best model (loss={best_loss:.6f})")

    return history
