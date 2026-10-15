"""Validation tests, written out so each one can be read and checked.

Discrimination  - does the model rank bad borrowers above good ones?
                  AUC, Gini (= 2*AUC - 1), Kolmogorov-Smirnov.
Calibration     - are the PD numbers themselves right?
                  Hosmer-Lemeshow, per-grade binomial test, Brier score.
Stability       - has the population moved since development?
                  Population Stability Index (PSI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


# --- discrimination -----------------------------------------------------------

def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    """Largest gap between the cumulative score distributions of bads and goods."""
    y, p = np.asarray(y), np.asarray(p)
    return float(stats.ks_2samp(p[y == 1], p[y == 0]).statistic)


def bootstrap_ci(y, p, metric=gini, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap interval. Resamples borrowers with replacement."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# --- calibration --------------------------------------------------------------

def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def hosmer_lemeshow(y, p, groups: int = 10) -> tuple[float, float]:
    """Chi-square on observed vs expected defaults across PD deciles.

    Small p-value = predicted PDs do not match observed default rates.
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["g"] = pd.qcut(df["p"].rank(method="first"), groups, labels=False)
    agg = df.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    agg["pbar"] = agg["exp"] / agg["n"]
    denom = agg["n"] * agg["pbar"] * (1 - agg["pbar"])
    stat = float((((agg["obs"] - agg["exp"]) ** 2) / denom.clip(lower=1e-12)).sum())
    return stat, float(stats.chi2.sf(stat, groups - 2))


def grade_table(y, p, edges=(0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.60, 1.0001)) -> pd.DataFrame:
    """Bucket borrowers into PD grades and test each grade.

    For each grade: is the observed default count consistent with the average
    PD assigned to it? Two-sided exact binomial test.
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    labels = [f"G{i + 1}" for i in range(len(edges) - 1)]
    df["grade"] = pd.cut(df["p"], bins=list(edges), labels=labels, right=False)
    rows = []
    for g, part in df.groupby("grade", observed=True):
        n, d, pd_mean = len(part), int(part["y"].sum()), float(part["p"].mean())
        rows.append({
            "grade": g,
            "pd_range": f"{edges[labels.index(g)]:.0%}-{min(edges[labels.index(g) + 1], 1):.0%}",
            "n": n, "defaults": d,
            "mean_pd": pd_mean, "observed_dr": d / n,
            "binomial_p": float(stats.binomtest(d, n, min(max(pd_mean, 1e-9), 1 - 1e-9)).pvalue),
        })
    out = pd.DataFrame(rows)
    out["result"] = np.where(out["binomial_p"] < 0.05, "FAIL", "pass")
    return out


# --- stability ----------------------------------------------------------------

def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` against `expected`.

    Rule of thumb: < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift.
    Bins are deciles of the expected sample.
    """
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float) -> str:
    return "stable" if value < 0.10 else "monitor" if value < 0.25 else "shifted"
