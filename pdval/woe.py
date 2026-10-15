"""Weight-of-evidence binning, the standard way credit scorecards are built.

Each variable is cut into a few bins; each bin is replaced by its WoE:

    WoE = ln( share of goods in bin / share of bads in bin )

Positive WoE means safer than average. Missing values get their own bin
rather than being imputed, because in credit data "not reported" is often
information in itself.

Information Value (IV) summarises how much a variable separates goods from
bads: below 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, above 0.3 strong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "Missing"
SMOOTH = 0.5          # avoids log(0) for bins with no goods or no bads
MIN_BIN_SHARE = 0.05  # bins smaller than this are merged with a neighbour


@dataclass
class VariableBins:
    name: str
    kind: str                        # "numeric" or "categorical"
    edges: list[float] = field(default_factory=list)
    table: pd.DataFrame | None = None

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def labels(self, series: pd.Series) -> pd.Series:
        if self.kind == "categorical":
            return series.astype("object").where(series.notna(), MISSING).astype(str)
        cut = pd.cut(series, bins=self.edges, include_lowest=True)
        return cut.astype(str).where(series.notna(), MISSING)

    def transform(self, series: pd.Series) -> pd.Series:
        woe = dict(zip(self.table["bin"], self.table["woe"]))
        # unseen categories score as neutral (WoE 0) rather than crashing
        return self.labels(series).map(woe).fillna(0.0).astype(float)


def _quantile_edges(values: pd.Series, max_bins: int) -> list[float]:
    qs = np.unique(np.quantile(values.dropna(), np.linspace(0, 1, max_bins + 1)))
    if len(qs) < 2:
        qs = np.array([values.min(), values.max()])
    edges = list(qs)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _table(labels: pd.Series, y: pd.Series) -> pd.DataFrame:
    grouped = pd.DataFrame({"bin": labels, "y": y}).groupby("bin", sort=False)["y"]
    t = pd.DataFrame({"n": grouped.size(), "bads": grouped.sum()}).reset_index()
    t["goods"] = t["n"] - t["bads"]
    total_g, total_b = t["goods"].sum(), t["bads"].sum()
    dist_g = (t["goods"] + SMOOTH) / (total_g + SMOOTH * len(t))
    dist_b = (t["bads"] + SMOOTH) / (total_b + SMOOTH * len(t))
    t["share"] = t["n"] / t["n"].sum()
    t["default_rate"] = t["bads"] / t["n"]
    t["woe"] = np.log(dist_g / dist_b)
    t["iv"] = (dist_g - dist_b) * t["woe"]
    return t


def fit_numeric(name: str, x: pd.Series, y: pd.Series, max_bins: int = 5) -> VariableBins:
    edges = _quantile_edges(x, max_bins)
    # merge thin bins into their left neighbour until every bin is big enough
    while len(edges) > 2:
        counts = pd.cut(x.dropna(), bins=edges, include_lowest=True).value_counts(sort=False)
        shares = counts.values / max(len(x), 1)
        thin = np.where(shares < MIN_BIN_SHARE)[0]
        if not len(thin):
            break
        i = int(thin[0])
        del edges[i if i > 0 else 1]
    vb = VariableBins(name, "numeric", edges)
    table = _table(vb.labels(x), y)
    order = [str(c) for c in pd.cut(x.dropna(), bins=edges, include_lowest=True).cat.categories]
    rank = {b: i for i, b in enumerate(order + [MISSING])}
    vb.table = table.sort_values("bin", key=lambda col: col.map(rank)).reset_index(drop=True)
    return vb


def fit_categorical(name: str, x: pd.Series, y: pd.Series) -> VariableBins:
    vb = VariableBins(name, "categorical")
    vb.table = _table(vb.labels(x), y)
    return vb


class WoEEncoder:
    """Fit bins on development data only, then apply the same bins anywhere."""

    def __init__(self, numeric: list[str], categorical: list[str], max_bins: int = 5):
        self.numeric, self.categorical, self.max_bins = numeric, categorical, max_bins
        self.bins: dict[str, VariableBins] = {}

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        for c in self.numeric:
            self.bins[c] = fit_numeric(c, df[c], y, self.max_bins)
        for c in self.categorical:
            self.bins[c] = fit_categorical(c, df[c], y)
        return self

    def transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or list(self.bins)
        return pd.DataFrame({c: self.bins[c].transform(df[c]) for c in cols}, index=df.index)

    def iv_table(self) -> pd.DataFrame:
        rows = [{"variable": k, "iv": v.iv, "bins": len(v.table)} for k, v in self.bins.items()]
        out = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
        out["strength"] = pd.cut(out["iv"], [-1, 0.02, 0.1, 0.3, 0.5, np.inf],
                                 labels=["useless", "weak", "medium", "strong", "suspicious"])
        return out
