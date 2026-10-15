# HMEQ data

`hmeq.csv`: 5,960 US home-equity loans, 12 applicant characteristics and the
target `BAD` (1 = defaulted or seriously delinquent, 19.9% of loans).

Public teaching dataset from Baesens, Roesch and Scheule, *Credit Risk
Analytics* (Wiley, 2016), also distributed by SAS. It contains no personal
data.

SHA-256: `dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa`
(checked by `pdval.data.load()` on every run).

| Column | Meaning |
|---|---|
| BAD | 1 = default or serious delinquency, 0 = repaid |
| LOAN | Amount of the loan request |
| MORTDUE | Amount due on existing mortgage |
| VALUE | Value of current property |
| REASON | DebtCon = debt consolidation, HomeImp = home improvement |
| JOB | Occupational category |
| YOJ | Years at present job |
| DEROG | Number of major derogatory reports |
| DELINQ | Number of delinquent credit lines |
| CLAGE | Age of oldest credit line in months |
| NINQ | Number of recent credit inquiries |
| CLNO | Number of credit lines |
| DEBTINC | Debt-to-income ratio |
