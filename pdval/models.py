"""Three models, so the validation has something to challenge.

champion   WoE logistic scorecard. The model a bank would actually put in
           front of a regulator: monotone, explainable, missing values binned.
legacy     A reconstruction of the 2023 notebook this repo replaces:
           median/mode imputation, scaling, one-hot, SMOTEENN resampling,
           logistic regression (the K-means segmentation step is left out).
challenger Gradient boosting on raw features. Used as a benchmark: if it is
           far better, the champion is leaving performance on the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, SEED, TARGET
from .woe import WoEEncoder

MIN_IV = 0.02
MAX_CORR = 0.7


@dataclass
class Champion:
    encoder: WoEEncoder | None = None
    selected: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    lr: LogisticRegression | None = None

    def fit(self, train: pd.DataFrame) -> "Champion":
        y = train[TARGET]
        self.encoder = WoEEncoder(NUMERIC, CATEGORICAL).fit(train, y)
        iv = self.encoder.iv_table().set_index("variable")["iv"]

        keep = [v for v in iv.index if iv[v] >= MIN_IV]
        for v in iv.index:
            if iv[v] < MIN_IV:
                self.dropped[v] = f"IV {iv[v]:.3f} below {MIN_IV}"

        # among highly correlated WoE pairs, keep the one with higher IV
        woe = self.encoder.transform(train, keep)
        corr = woe.corr().abs()
        for a in list(keep):
            for b in list(keep):
                if a != b and a in keep and b in keep and corr.loc[a, b] > MAX_CORR:
                    loser = a if iv[a] < iv[b] else b
                    keep.remove(loser)
                    self.dropped[loser] = f"correlation {corr.loc[a, b]:.2f} with {b if loser == a else a}"

        self.selected = keep
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.lr.fit(self.encoder.transform(train, keep), y)
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.encoder.transform(df, self.selected))[:, 1]

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"variable": self.selected, "coefficient": self.lr.coef_[0]})


def _preprocess(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


@dataclass
class Legacy:
    prep: ColumnTransformer | None = None
    lr: LogisticRegression | None = None

    def fit(self, train: pd.DataFrame) -> "Legacy":
        self.prep = _preprocess(scale=True)
        X = self.prep.fit_transform(train)
        Xs, ys = SMOTEENN(random_state=SEED).fit_resample(X, train[TARGET])
        self.lr = LogisticRegression(max_iter=1000).fit(Xs, ys)
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.prep.transform(df))[:, 1]


@dataclass
class Challenger:
    model: Pipeline | None = None

    def fit(self, train: pd.DataFrame) -> "Challenger":
        prep = ColumnTransformer([
            ("num", "passthrough", NUMERIC),   # boosting handles NaN natively
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ], sparse_threshold=0)
        self.model = Pipeline([
            ("prep", prep),
            ("gbm", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                   max_iter=300, random_state=SEED)),
        ]).fit(train, train[TARGET])
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df)[:, 1]
