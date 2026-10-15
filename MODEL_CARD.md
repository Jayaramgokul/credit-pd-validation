# Model card: HMEQ PD scorecard (Champion)

| | |
|---|---|
| **Purpose** | Estimate the probability that a home-equity borrower defaults or becomes seriously delinquent (`BAD = 1`) |
| **Type** | Logistic regression on weight-of-evidence (WoE) transformed variables |
| **Owner** | Jayaram Gokulan (portfolio project; not used for real lending) |
| **Status** | Not approved: see findings F2 and F3 in [VALIDATION_REPORT.md](VALIDATION_REPORT.md) |

## Intended use

Teaching and demonstration of PD model development and independent validation.
**Not** for credit decisions, provisioning or capital calculations.

## Data

HMEQ public dataset, 5,960 loans, 19.9% default rate, 12 characteristics (see
[data/README.md](data/README.md)). Stratified 70/30 development and hold-out
split, seed 42. No dates, so no out-of-time sample.

## Method

1. Each numeric variable is cut into up to five quantile bins, thin bins
   (below 5% of loans) are merged, and missing values form their own bin.
   Categorical variables use their categories plus a missing bin.
2. Every bin is replaced by its WoE, fitted on development data only.
3. Variables with information value below 0.02 are dropped; of any pair with
   WoE correlation above 0.7, the weaker is dropped.
4. Unpenalised logistic regression on the remaining WoE variables.
5. No rebalancing, so predicted PDs stay on the observed base rate.

## Performance (hold-out)

Gini 0.778 (95% bootstrap interval 0.742 to 0.811), KS 0.61, mean PD 20.3%
against an observed default rate of 20.0%. Full results, including
calibration by grade, are in the validation report.

## Known limitations

- Depends heavily on whether DEBTINC is missing; the reason for that
  missingness is unknown and could be target leakage (finding F2).
- Under-predicts default in the 5-10% PD grade (finding F3).
- A gradient-boosting benchmark discriminates better (finding F4).
- Trained on a small, old, US-only teaching dataset; not representative of any
  current portfolio.
