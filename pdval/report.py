"""Run the full validation and write VALIDATION_REPORT.md plus figures.

    python -m pdval.report

Every number in the report is computed here, on every run. Nothing is
typed in by hand, so the report cannot drift from the code.
"""

from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import data, metrics as m  # noqa: E402
from .models import Challenger, Champion, Legacy  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

# Reference palette, slots 1-3 (validated for all-pairs use in light mode).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Champion": "#2a78d6", "Legacy (2023)": "#eb6834", "Challenger": "#1baf7a"}


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f(v) if f else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


pct = lambda v: "n/a" if pd.isna(v) else f"{v:.1f}%"   # noqa: E731
pct1 = lambda v: f"{100 * v:.1f}%"                      # noqa: E731


def run(n_boot: int = 1000) -> dict:
    df = data.load()
    s = data.split(df)
    ytr, yte = s.train[data.TARGET].values, s.test[data.TARGET].values

    models = {"Champion": Champion().fit(s.train),
              "Legacy (2023)": Legacy().fit(s.train),
              "Challenger": Challenger().fit(s.train)}
    preds = {k: (v.predict_pd(s.train), v.predict_pd(s.test)) for k, v in models.items()}

    rows = []
    for name, (ptr, pte) in preds.items():
        lo, hi = m.bootstrap_ci(yte, pte, n=n_boot)
        hl, hl_p = m.hosmer_lemeshow(yte, pte)
        grades = m.grade_table(yte, pte)
        rows.append({
            "model": name,
            "gini_train": m.gini(ytr, ptr), "gini_test": m.gini(yte, pte),
            "gini_ci": f"{lo:.3f} to {hi:.3f}", "ks_test": m.ks(yte, pte),
            "mean_pd": pte.mean(), "observed_dr": yte.mean(),
            "brier": m.brier(yte, pte), "hl_p": hl_p,
            "grades_failed": f"{(grades.result == 'FAIL').sum()} of {len(grades)}",
            "psi_score": m.psi(ptr, pte),
        })
    summary = pd.DataFrame(rows)

    champ: Champion = models["Champion"]
    iv = champ.encoder.iv_table()
    dq = data.data_quality(df)
    coefs = champ.coefficients()

    # Sensitivity: refit the champion's regression without DEBTINC, same bins.
    without = [v for v in champ.selected if v != "DEBTINC"]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(champ.encoder.transform(s.train, without), ytr)
    gini_wo_debtinc = m.gini(yte, lr.predict_proba(champ.encoder.transform(s.test, without))[:, 1])

    char_psi = pd.DataFrame([
        {"variable": v, "psi": m.psi(champ.encoder.bins[v].transform(s.train[v]),
                                     champ.encoder.bins[v].transform(s.test[v]))}
        for v in champ.selected])
    char_psi["status"] = char_psi["psi"].map(m.psi_label)

    legacy = models["Legacy (2023)"]
    from imblearn.combine import SMOTEENN
    _, ys = SMOTEENN(random_state=data.SEED).fit_resample(legacy.prep.transform(s.train), ytr)
    resampled_bad_share = float(np.mean(ys))

    ci_lo_cg = m.bootstrap_ci(yte, preds["Challenger"][1], n=n_boot)[0]
    ci_hi_ch = m.bootstrap_ci(yte, preds["Champion"][1], n=n_boot)[1]

    out = dict(ci_lo_cg=ci_lo_cg, ci_hi_ch=ci_hi_ch, resampled_bad_share=resampled_bad_share, df=df, split=s, models=models, preds=preds, summary=summary, iv=iv, dq=dq,
               coefs=coefs, gini_wo_debtinc=gini_wo_debtinc, char_psi=char_psi,
               grades={k: m.grade_table(yte, v[1]) for k, v in preds.items()})
    figures(out)
    write_report(out)
    return out


def figures(r: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    yte = r["split"].test[data.TARGET].values

    # ROC
    fig, ax = plt.subplots(figsize=(6.4, 5), dpi=160)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    for name, (_, pte) in r["preds"].items():
        fpr, tpr, _ = roc_curve(yte, pte)
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2,
                label=f"{name}  Gini {m.gini(yte, pte):.2f}")
    _style(ax, "Discrimination on the hold-out sample (ROC)")
    ax.set_xlabel("False positive rate (goods flagged)")
    ax.set_ylabel("True positive rate (bads caught)")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "roc.png")
    plt.close(fig)

    # Calibration by decile
    fig, ax = plt.subplots(figsize=(6.4, 5), dpi=160)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    ax.text(0.62, 0.66, "perfect calibration", color=INK2, fontsize=8, rotation=38)
    for name, (_, pte) in r["preds"].items():
        d = pd.DataFrame({"y": yte, "p": pte})
        d["dec"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False)
        g = d.groupby("dec").agg(pred=("p", "mean"), obs=("y", "mean"))
        ax.plot(g["pred"], g["obs"], color=COLORS[name], linewidth=2, marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    _style(ax, "Calibration: predicted PD vs observed default rate")
    ax.set_xlabel("Mean predicted PD (decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)

    # DEBTINC missingness
    dq = r["dq"].dropna(subset=["default_rate_when_missing"])
    dq = dq.assign(gap=(dq.default_rate_when_missing - dq.default_rate_when_present).abs())
    dq = dq.sort_values("gap", ascending=False).head(6).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    y = np.arange(len(dq))
    h = 0.36
    ax.barh(y + h / 2, dq["default_rate_when_missing"].fillna(0), height=h - 0.04,
            color="#eb6834", label="value missing")
    ax.barh(y - h / 2, dq["default_rate_when_present"], height=h - 0.04,
            color="#2a78d6", label="value present")
    for yi, a, b in zip(y, dq["default_rate_when_missing"], dq["default_rate_when_present"]):
        ax.text(a + 1, yi + h / 2, f"{a:.0f}%", va="center", fontsize=8, color=INK)
        ax.text(b + 1, yi - h / 2, f"{b:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, dq["variable"])
    _style(ax, "Default rate when a field is missing vs present\n(six largest gaps)")
    ax.set_xlabel("Default rate (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "missingness.png")
    plt.close(fig)


def write_report(r: dict) -> None:
    s, summ, iv, dq = r["split"], r["summary"], r["iv"], r["dq"]
    get = lambda name, col: summ.set_index("model").loc[name, col]   # noqa: E731
    deb = dq.set_index("variable").loc["DEBTINC"]
    val = dq.set_index("variable").loc["VALUE"]
    g_ch, g_cg, g_lg = get("Champion", "gini_test"), get("Challenger", "gini_test"), get("Legacy (2023)", "gini_test")
    pd_lg, dr = get("Legacy (2023)", "mean_pd"), get("Legacy (2023)", "observed_dr")
    champ_grades = r["grades"]["Champion"]
    under = champ_grades[(champ_grades.result == "FAIL") & (champ_grades.observed_dr > champ_grades.mean_pd)]
    debt_iv = float(iv.set_index("variable").loc["DEBTINC", "iv"])

    summary_view = summ.rename(columns={
        "gini_train": "Gini (train)", "gini_test": "Gini (test)", "gini_ci": "Gini 95% CI",
        "ks_test": "KS", "mean_pd": "Mean PD", "observed_dr": "Observed DR",
        "brier": "Brier", "hl_p": "HL p-value", "grades_failed": "Grades failing",
        "psi_score": "Score PSI"})

    lines = [
        "# Independent validation report: HMEQ probability-of-default model",
        "",
        f"*Generated by `python -m pdval.report` on {dt.date.today():%d %B %Y}. "
        "Every figure below is recomputed on each run.*",
        "",
        "## 1. Scope",
        "",
        "Model under review: a probability-of-default (PD) scorecard for home-equity loans, built as a "
        "weight-of-evidence (WoE) logistic regression (**Champion**). Two comparators are assessed alongside it:",
        "",
        "- **Legacy (2023):** a reconstruction of my earlier notebook approach (median imputation, "
        "SMOTEENN rebalancing, logistic regression).",
        "- **Challenger:** gradient boosting on raw features, used purely as a performance benchmark.",
        "",
        f"Data: {len(r['df']):,} loans, {100 * r['df'].BAD.mean():.1f}% default rate. "
        f"Stratified split: {len(s.train):,} development / {len(s.test):,} hold-out (seed {data.SEED}). "
        "Tests follow the usual structure of model risk validation (SR 11-7 / PRA SS1/23): "
        "data, conceptual soundness, discrimination, calibration, stability, benchmarking.",
        "",
        "## 2. Summary of results",
        "",
        _table(summary_view, {"Mean PD": pct1, "Observed DR": pct1,
                              "HL p-value": lambda v: f"{v:.3f}" if v >= 0.001 else "<0.001"}),
        "",
        "![ROC](reports/figures/roc.png)",
        "![Calibration](reports/figures/calibration.png)",
        "",
        "## 3. Findings",
        "",
        "Severity uses a common three-level scale: **High** must be fixed before use, **Medium** "
        "needs a remediation plan, **Low** is an observation.",
        "",
        "### F1 (High): Rebalancing destroys PD calibration in the legacy model",
        "",
        f"After SMOTEENN the training data is {pct1(r['resampled_bad_share'])} bad, against "
        f"{pct1(r['df'].BAD.mean())} in reality, so the model learns the wrong base rate. On the hold-out sample the legacy model's average PD is **{pct1(pd_lg)}** against an "
        f"observed default rate of **{pct1(dr)}**. "
        f"{r['grades']['Legacy (2023)'].result.eq('FAIL').sum()} of "
        f"{len(r['grades']['Legacy (2023)'])} PD grades fail the binomial test, and Hosmer-Lemeshow rejects "
        "calibration outright. Rebalancing can help a classifier pick a threshold, but a PD model's output "
        "is a probability that feeds provisions and capital, so it must stay on the true base rate. "
        "**Remediation:** do not rebalance PD models; if rebalancing is unavoidable, recalibrate to the "
        "observed default rate afterwards. Applied in the Champion.",
        "",
        "### F2 (High): Missing values carry the strongest signal, and the reason is unexplained",
        "",
        f"When DEBTINC is missing ({deb.missing_pct:.1f}% of loans) the default rate is "
        f"**{deb.default_rate_when_missing:.1f}%**, against {deb.default_rate_when_present:.1f}% when it is "
        f"present. For VALUE the gap is {val.default_rate_when_missing:.1f}% vs "
        f"{val.default_rate_when_present:.1f}% ({val.missing_pct:.1f}% missing). Median imputation, as used "
        f"in the legacy model, erases this, which is a large part of why its Gini is only {g_lg:.2f}. "
        f"The Champion keeps missing values as their own WoE bin, and DEBTINC's information value is "
        f"{debt_iv:.2f}, far above the 0.5 level at which a variable is normally treated as suspiciously "
        f"strong. Without DEBTINC the Champion's Gini falls from {g_ch:.3f} to {r['gini_wo_debtinc']:.3f}. "
        "If DEBTINC goes missing *because of* the outcome (for example, it is only collected for "
        "applications that progress normally), this is target leakage and the model would not perform "
        "like this in production. **Remediation:** the model owner must document why DEBTINC is missing "
        "before the variable is accepted.",
        "",
        "![Missingness](reports/figures/missingness.png)",
        "",
        "### F3 (Medium): Champion is accurate on average but mis-calibrated in the middle grades",
        "",
        f"Average PD matches the observed rate ({pct1(get('Champion', 'mean_pd'))} vs "
        f"{pct1(get('Champion', 'observed_dr'))}), but Hosmer-Lemeshow rejects decile-level calibration "
        f"and {(champ_grades.result == 'FAIL').sum()} of {len(champ_grades)} grades fail the binomial test. "
        + (f"The concern is grade {', '.join(under.grade.astype(str))}, where the observed default rate "
           f"({', '.join(pct1(v) for v in under.observed_dr)}) is above the assigned PD "
           f"({', '.join(pct1(v) for v in under.mean_pd)}): the model is *not conservative* there. "
           if len(under) else "")
        + "Failures where the model over-predicts are conservative and lower priority. "
        "**Remediation:** recalibrate by grade (for example isotonic or a grade-level scaling) and add a "
        "margin of conservatism.",
        "",
        _table(champ_grades.assign(mean_pd=champ_grades.mean_pd.map(pct1),
                                   observed_dr=champ_grades.observed_dr.map(pct1),
                                   binomial_p=champ_grades.binomial_p.map(lambda v: f"{v:.3f}"))),
        "",
        "### F4 (Medium): The benchmark is materially stronger",
        "",
        f"The Challenger reaches a hold-out Gini of **{g_cg:.3f}** against the Champion's {g_ch:.3f}. "
        + ("It is also better calibrated (lower Brier score). " if get("Challenger", "brier") < get("Champion", "brier") else "")
        + f"The bootstrap intervals "
        f"({get('Challenger', 'gini_ci')} vs {get('Champion', 'gini_ci')}) "
        + ("do not overlap, so the gap is not noise. " if r["ci_lo_cg"] > r["ci_hi_ch"] else "overlap, so the gap may be noise. ")
        + "Its train/test gap "
        f"({get('Challenger', 'gini_train'):.3f} vs {g_cg:.3f}) shows some overfitting. A gap this size "
        "usually means interactions the scorecard cannot express. **Remediation:** investigate which "
        "interactions drive the lift (e.g. SHAP on the Challenger), and either add them to the scorecard "
        "or record why interpretability is worth the lost discrimination.",
        "",
        "### F5 (Low): Variable review",
        "",
        ("All selected WoE coefficients are negative, which is the correct sign (higher WoE means safer). "
         if (r["coefs"].coefficient < 0).all() else
         f"Coefficients with the wrong sign: {', '.join(r['coefs'][r['coefs'].coefficient >= 0].variable)}. ")
        + 
        f"Dropped: {', '.join(f'{k} ({v})' for k, v in r['models']['Champion'].dropped.items()) or 'none'}. "
        f"Weak variables kept in the model ({', '.join(iv[(iv.iv < 0.1) & iv.variable.isin(r['models']['Champion'].selected)].variable)}) "
        "add complexity for little gain and could be pruned.",
        "",
        _table(iv.merge(r["coefs"], on="variable", how="left").rename(
            columns={"iv": "IV", "bins": "Bins", "strength": "Strength", "coefficient": "Coefficient"}),
            {"Coefficient": lambda v: "dropped" if pd.isna(v) else f"{v:.3f}"}),
        "",
        "### F6 (Low): Stability can only be checked weakly",
        "",
        f"Score PSI between development and hold-out is {get('Champion', 'psi_score'):.4f}, and the "
        f"largest characteristic PSI is {r['char_psi'].psi.max():.4f} ({r['char_psi'].sort_values('psi').variable.iloc[-1]}), "
        "all stable, but that is expected from a random split. HMEQ has no dates, so a true "
        "out-of-time test is impossible. **Remediation:** before use, test on a later vintage and set "
        "monitoring triggers (PSI above 0.10 = review, above 0.25 = redevelop).",
        "",
        "## 4. Overall opinion",
        "",
        "The Champion is conceptually sound and a clear improvement on the 2023 approach, but it is "
        "**not fit for use** until F2 (the source of DEBTINC missingness) is explained, and its "
        "middle-grade calibration (F3) is fixed. The Legacy model should be retired.",
        "",
        "## 5. Limitations of this validation",
        "",
        "- Public teaching dataset; no business owner to answer F2, no production data, no vintages.",
        "- Single random split; results could move with a different seed (bootstrap intervals give a sense of how far).",
        "- The Legacy reconstruction leaves out the K-means segmentation step of the original notebook. "
        "The original also imputed missing values on the full dataset *before* splitting, which leaks "
        "hold-out information; the reconstruction fits imputation on the development sample only.",
        "",
    ]
    (ROOT / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    res = run()
    print(res["summary"].round(3).to_string(index=False))
    print(f"\nwrote {ROOT / 'VALIDATION_REPORT.md'}")
