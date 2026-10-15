"""Load the HMEQ data, check it is what we think it is, and split it.

HMEQ: 5,960 US home-equity loans, target BAD = 1 if the borrower defaulted
or was seriously delinquent. Public dataset from Baesens, Roesch & Scheule,
"Credit Risk Analytics" (Wiley, 2016); also distributed by SAS.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hmeq.csv"
EXPECTED_SHA256 = "dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa"

TARGET = "BAD"
NUMERIC = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ",
           "CLAGE", "NINQ", "CLNO", "DEBTINC"]
CATEGORICAL = ["REASON", "JOB"]
FEATURES = NUMERIC + CATEGORICAL

DESCRIPTIONS = {
    "LOAN": "Amount of the loan request",
    "MORTDUE": "Amount due on existing mortgage",
    "VALUE": "Value of current property",
    "REASON": "DebtCon = debt consolidation, HomeImp = home improvement",
    "JOB": "Occupational category",
    "YOJ": "Years at present job",
    "DEROG": "Number of major derogatory reports",
    "DELINQ": "Number of delinquent credit lines",
    "CLAGE": "Age of oldest credit line in months",
    "NINQ": "Number of recent credit inquiries",
    "CLNO": "Number of credit lines",
    "DEBTINC": "Debt-to-income ratio",
}

SEED = 42


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path = DATA_PATH, verify: bool = True) -> pd.DataFrame:
    """Read the CSV and fail loudly if it is not the expected file."""
    if verify and sha256(path) != EXPECTED_SHA256:
        raise ValueError(f"{path} does not match the expected HMEQ checksum")
    df = pd.read_csv(path)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"columns missing from data: {sorted(missing_cols)}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("target BAD must be 0/1")
    return df


def split(df: pd.DataFrame, test_size: float = 0.3, seed: int = SEED) -> Split:
    """Stratified development / hold-out split.

    HMEQ has no dates, so a true out-of-time sample is impossible. This is a
    stated limitation of the validation, not something to paper over.
    """
    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET],
                                   random_state=seed)
    return Split(train.reset_index(drop=True), test.reset_index(drop=True))


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Missing share and default rate with / without each value present."""
    rows = []
    for col in FEATURES:
        miss = df[col].isna()
        rows.append({
            "variable": col,
            "missing_pct": 100 * miss.mean(),
            "default_rate_when_missing": 100 * df.loc[miss, TARGET].mean() if miss.any() else float("nan"),
            "default_rate_when_present": 100 * df.loc[~miss, TARGET].mean(),
        })
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)
