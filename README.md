# IEEE-CIS Fraud Detection

Predicting fraudulent e-commerce transactions on real payment data from Vesta Corporation ([Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection)).

This project is less about which algorithm I picked and more about **how I made decisions under uncertainty**: masked features with no documentation, a hard time gap between train and test, severe class imbalance, and a CPU-only compute budget.

## Results at a Glance

| Metric | Score |
|--------|-------|
| Cross-validated AUC (OOF, final ensemble) | **0.93985** |
| Public Leaderboard | **0.944425** |
| Private Leaderboard | **0.922207** |
| Precision / Recall / F1 (F1-optimal threshold) | 0.7248 / 0.5624 / 0.6333 |

## How I Framed the Problem

- **Accuracy is useless here.** Only 3.5% of transactions are fraud, so predicting "never fraud" scores 96.5% accuracy while catching nothing. I optimised **ROC-AUC** (threshold-free, rank-based) and also tracked **PR-AUC**, which is far more honest about the rare class (~0.63–0.65 vs ~0.93 ROC-AUC).
- **Fraud belongs to a client, not a transaction.** A stolen card is used repeatedly by the same person. The dataset never says which transactions belong to whom, so reconstructing that became the central idea of the project.
- **Train and test don't overlap in time** (days 0–183 vs 213–396). Any validation that ignores this will lie.

## My Thought Process

### 1. Reverse-engineer what the data can't tell me
Most columns (C, D, M, V, id) are masked with no data dictionary, so I learned what they meant from how they behaved:
- **D columns** split into "age-type" (grow with calendar time) and "gap-type" (reset constantly). D9 turned out to be simply *hour ÷ 24*, so it was dropped.
- **Missingness is a signal, not noise.** Fraud rate is **7.85%** when identity data exists vs **2.09%** when it doesn't. So I never mean-imputed; tree models learn where missing values should go.
- **Quiet hours are risky hours.** Fraud jumps above **10%** during low-traffic hours vs ~3% at peak. Criminals don't sleep; genuine shoppers do. Weekday showed no signal, so it was tested and dropped.
- **Email domains** were split into provider and country suffix, then grouped by parent company, except **protonmail**, which was kept separate because it was the strongest single signal and grouping would have diluted it.
- **Caught a confound in my own analysis:** a match-flag feature suggested "partially verified" transactions were riskiest (6.56%). Holding product type fixed completely reversed the pattern. Lesson: control for known drivers before trusting subgroup comparisons.

### 2. Rebuild the client identity (UID)
```
UID = card1 + addr1 + floor(day − D1)
```
Subtracting D1 from the day turns a drifting "days since card first seen" into a fixed anchor, so the same card maps to the same ID across days. **97.8%** of multi-transaction UIDs had fully consistent labels, confirming it isolates real clients.

On top of the UID I built aggregates like the transaction's amount relative to that client's usual spend (ratio, z-score, percentile), time-of-day consistency, match-flag rates and device diversity. **Fraud shows up as a deviation from a client's own normal**, and trees can't compute ratios on their own, so these had to be engineered explicitly.

The UID itself was **never used as a feature**, only its aggregates. Many test clients never appear in train, so memorising IDs teaches nothing transferable.

### 3. Build validation before building any model
| Option | Decision | Why |
|--------|----------|-----|
| Random / Stratified KFold | Rejected | Lets the model learn from the future; inflated, meaningless CV |
| TimeSeriesSplit | Rejected | Time-pure, but early folds train on ~1 month and are noisy |
| **GroupKFold (month as group, 6 folds)** | **Chosen** | Every fold trains on ~5 months; blocks entity leakage, the dominant risk here |

I added a second check, **Split B** (train months 0–4, validate on month 5), which mimics the real test setup most closely, with a rule set in advance: *if the two splits disagree, investigate, don't pick blindly.* Month 6 (only 8k rows) was merged into month 5, and every fold was asserted to have no month on both sides.

### 4. Diagnose drift with adversarial validation
I trained a classifier to answer "is this row from train or test?". Its AUC went **1.000 → 0.881** as I fixed issues, and its feature importances pointed straight at problems:
- **Caught a silent bug:** my "detrended" D columns had train means around −50 and test means around +150. The fix had made drift worse, so the five broken columns were dropped.
- **Overturned my own hypothesis:** I expected C columns to be running totals (higher in test). They were actually *lower* in test, because test contains more new clients with less history. That's a population difference, not time drift, so detrending would have been the wrong fix.
- **Knew when to stop.** Remaining drift (e.g. browser versions rising over time) was real and explainable. Deleting predictive features to push a diagnostic number toward 0.5 would have cost real fraud signal.

### 5. Choose models that fit the shape of the data
The data is tabular, mixed-type, heavily missing, non-monotonic and interaction-heavy, which is exactly what gradient-boosted trees handle natively. I rejected Logistic Regression (can't model non-monotonic patterns without imputing away the missingness signal), SVM and KNN (don't scale / no meaningful distance across 324 mixed columns), and deep learning (mismatched inductive bias for rule-like data, high cost for no expected gain).

Instead of one "best" model, I trained **three boosters with different tree-growth strategies** (XGBoost level-wise, LightGBM leaf-wise, CatBoost symmetric) so they would make *different mistakes*. Each got its own prepared copy of one master dataset, since each needs different NaN and categorical handling. Encoders were fit once on train + test features combined to avoid silent category mismatches.

### 6. Deliberately skip SMOTE
- AUC is rank-based, so rebalancing barely changes what the metric measures.
- SMOTE interpolates between points, but the midpoint between card ID 13926 and 2755 is a card that doesn't exist. It would manufacture impossible transactions.
- Undersampling would throw away ~549,000 legitimate rows, which is how the model learns what "normal" looks like.

### 7. Spend compute where it pays off
One 6-fold run cost 18–58 minutes on CPU. A 20-iteration random search would take 6+ hours for an expected gain of ~0.002, while feature engineering moves AUC by 0.01+. So I skipped formal tuning, used reasoned parameters adapted from the winning solution, relied on early stopping, and invested the time in ensembling and post-processing.

### 8. Ensemble based on evidence, then hedge
Out-of-fold prediction correlations were 0.945 (LGB–XGB), 0.903 (CAT–LGB) and 0.868 (CAT–XGB): diverse enough to justify combining.
- **Rank averaging**, so no model dominates just because of its score scale.
- **Stacking** with a Logistic Regression meta-model on time-respecting OOF predictions (simple on purpose: 3 inputs don't need a powerful learner).
- **The two validation splits disagreed** (Split A preferred blending, Split B preferred stacking), so I averaged both 50/50. That hedge scored best on Split A and matched the best on Split B.

### 9. Add back what the model can't see
Predictions were smoothed within each UID, since a client's transactions should broadly agree. Full smoothing *hurt* (0.93332 → 0.93235), but sweeping the strength showed partial smoothing at **α = 0.5** gave **+0.0017** (→ 0.93502). Testing a range instead of one extreme turned "doesn't work" into a real gain.

## Model Results

| Model | OOF AUC | PR-AUC | Train Time |
|-------|---------|--------|-----------|
| LightGBM | 0.93221 | 0.648 | 18 min |
| XGBoost | 0.93008 | 0.634 | 47 min |
| CatBoost | 0.92411 | 0.627 | 58 min |

| Ensemble Step | OOF AUC |
|---------------|---------|
| Best single model (LightGBM) | 0.93221 |
| 2-model rank blend (LGB + XGB) | 0.93719 |
| 3-model rank blend | 0.93856 |
| 50/50 blend + stack hedge | **0.93985** |

- **CatBoost was the weakest model but still improved the ensemble** (+0.00137), because it was the least correlated. Ensembling is about collecting *differently wrong* models, not just good ones.
- **All three models scored worst on the same fold** (month 0), which has a much lower fraud rate (2.48%). Three algorithms agreeing points to the data, not the models.
- **LightGBM was 2.6× faster than XGBoost with a better score**, while XGBoost was more stable across folds: a genuine trade-off.


## Tech Stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · CatBoost · Matplotlib · Seaborn
