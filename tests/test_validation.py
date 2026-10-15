"""Known-answer tests for the validation metrics.

A validator's tools need validating too: each test pins a metric to a case
whose correct answer is known in advance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdval import data, metrics as m  # noqa: E402
from pdval.models import Champion  # noqa: E402
from pdval.woe import MISSING, WoEEncoder, fit_categorical, fit_numeric  # noqa: E402

rng = np.random.default_rng(7)


# --- discrimination ---------------------------------------------------------

def test_perfect_ranking_has_gini_one_and_ks_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert m.gini(y, p) == pytest.approx(1.0)
    assert m.ks(y, p) == pytest.approx(1.0)


def test_random_scores_have_gini_near_zero():
    y = rng.integers(0, 2, 20000)
    assert abs(m.gini(y, rng.random(20000))) < 0.03


def test_gini_matches_sklearn_auc():
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    assert m.gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)


def test_bootstrap_interval_contains_the_point_estimate():
    y = rng.integers(0, 2, 800)
    p = np.clip(y * 0.4 + rng.random(800) * 0.6, 0, 1)
    lo, hi = m.bootstrap_ci(y, p, n=200)
    assert lo < m.gini(y, p) < hi


# --- calibration ------------------------------------------------------------

def test_hosmer_lemeshow_accepts_true_probabilities():
    p = rng.uniform(0.02, 0.6, 20000)
    y = (rng.random(20000) < p).astype(int)
    _, pval = m.hosmer_lemeshow(y, p)
    assert pval > 0.01


def test_hosmer_lemeshow_rejects_doubled_probabilities():
    p = rng.uniform(0.02, 0.4, 20000)
    y = (rng.random(20000) < p).astype(int)
    _, pval = m.hosmer_lemeshow(y, np.clip(2 * p, 0, 0.99))
    assert pval < 0.001


def test_grade_table_flags_a_miscalibrated_grade():
    p = np.full(2000, 0.05)
    y = (rng.random(2000) < 0.15).astype(int)      # true rate 3x the PD
    table = m.grade_table(y, p)
    assert (table.result == "FAIL").all()


def test_brier_of_perfect_forecast_is_zero():
    assert m.brier([0, 1, 1], [0.0, 1.0, 1.0]) == 0.0


# --- stability --------------------------------------------------------------

def test_psi_is_zero_for_identical_samples():
    x = rng.normal(size=5000)
    assert m.psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_shift():
    x = rng.normal(size=5000)
    assert m.psi(x, x + 1.0) > 0.25
    assert m.psi_label(m.psi(x, x + 1.0)) == "shifted"


# --- weight of evidence ----------------------------------------------------

def test_woe_sign_follows_risk():
    # higher x = higher default rate, so the top bin must have negative WoE
    x = pd.Series(rng.random(4000))
    y = pd.Series((rng.random(4000) < 0.05 + 0.4 * x).astype(int))
    bins = fit_numeric("x", x, y)
    woe = bins.table.set_index("bin")["woe"]
    assert woe.iloc[0] > 0 > woe.iloc[-1]


def test_missing_values_get_their_own_bin():
    x = pd.Series([1.0, 2.0, np.nan, 3.0, np.nan] * 200)
    y = pd.Series([0, 0, 1, 0, 1] * 200)
    bins = fit_numeric("x", x, y)
    assert MISSING in set(bins.table["bin"])
    assert bins.transform(pd.Series([np.nan])).iloc[0] < 0     # missing = risky here


def test_unseen_category_scores_neutral():
    bins = fit_categorical("c", pd.Series(["a", "b"] * 50), pd.Series([0, 1] * 50))
    assert bins.transform(pd.Series(["zzz"])).iloc[0] == 0.0


def test_information_value_is_non_negative():
    df = data.load()
    enc = WoEEncoder(data.NUMERIC, data.CATEGORICAL).fit(df, df[data.TARGET])
    assert (enc.iv_table()["iv"] >= 0).all()


# --- data and model ----------------------------------------------------------

def test_data_checksum_and_shape():
    df = data.load()
    assert df.shape == (5960, 13)
    assert df[data.TARGET].mean() == pytest.approx(0.1995, abs=1e-3)


def test_split_is_stratified_and_disjoint():
    s = data.split(data.load())
    assert len(s.train) + len(s.test) == 5960
    assert s.train[data.TARGET].mean() == pytest.approx(s.test[data.TARGET].mean(), abs=0.005)


def test_champion_fits_and_stays_on_base_rate():
    s = data.split(data.load())
    model = Champion().fit(s.train)
    p = model.predict_pd(s.test)
    assert 0 <= p.min() and p.max() <= 1
    assert abs(p.mean() - s.test[data.TARGET].mean()) < 0.02
    assert m.gini(s.test[data.TARGET], p) > 0.6
