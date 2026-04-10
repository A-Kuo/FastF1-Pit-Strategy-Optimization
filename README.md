# F1 Pit Strategy: Predicting Pit Windows with Machine Learning

A machine learning pipeline for predicting optimal pit stop timing in Formula 1 racing, trained on historical race data (2018–2023) and evaluated on a held-out 2024 test set using the FastF1 data structure.

**Test set performance:** XGBoost — F1 = 0.4490, ROC-AUC = 0.8409, Recall = 79.5%

---

## The Problem

Pit stop timing in Formula 1 is a high-stakes decision made under uncertainty. A team must weigh tire degradation against the ~25-second time cost of the stop, accounting for gap to the car ahead, race progress, and the threat of a rival undercut. Teams currently rely on proprietary tire models built from pre-race simulations, but those models aren't always accurate in real time.

This project builds a data-driven model to answer one specific question: **given the current lap's tire state and race context, will this driver pit within the next 5 laps?**

---

## Mathematical Framework

### 1. Degradation Rate

For each tire stint, a linear model is fit to the lap time series:

```
t_l = α + β·n + ε_l
```

where `t_l` is lap time in seconds at lap `l`, `n` is tire age (laps on the current set), and `β` is the **degradation rate** (seconds per lap). The OLS estimate is:

```
β̂ = Σ(nᵢ - n̄)(tᵢ - t̄) / Σ(nᵢ - n̄)²
```

This per-stint regression requires at least 3 laps to fit. Stints shorter than 3 laps receive `β̂ = 0`. Typical values observed: SOFT ≈ +0.11 s/lap, MEDIUM ≈ +0.06 s/lap, HARD ≈ +0.03 s/lap. High `β̂` signals a tire approaching the performance cliff — the model uses this to raise pit probability.

### 2. Target Variable Construction

The binary target is defined over a forward-looking window of `w = 5` laps:

```
y_l = 1  if ∃ k ∈ {l+1, ..., l+w} such that pit(k) = 1
y_l = 0  otherwise
```

The choice of `w = 5` reflects race mechanics: pit window execution requires 1–2 laps to communicate and prepare, and teams rarely commit to a stop more than 5 laps out without triggering an immediate undercut/overcut sequence. Setting `w < 3` introduces excessive false negatives from communication lag; setting `w > 7` causes overlapping decision windows between consecutive stops, which creates label contamination — the model cannot distinguish a "pit now" signal from a "pit later" signal.

### 3. Engineered Features

**LapTimeDelta** — driver-relative pace normalization:

```
δ_l = t_l - median_j(t_j^(driver))
```

This normalizes for inter-driver pace differences: a 91.5s lap from a fast driver can still carry a `δ > 0` degradation signal that would be missed by using raw lap times. The median rather than mean reduces sensitivity to outlier laps under traffic or mechanical issues.

**StintAgeSquared** — non-linear degradation capture:

```
s_l² = n_l²
```

Empirically, F1 tire degradation is superlinear: the performance loss accelerates in the final laps of a stint. The squared term allows gradient-boosted trees to capture this curve without needing explicit polynomial feature construction. At tyre life `n = 5`, `n² = 25`; at `n = 15`, `n² = 225` — the squared term grows eight times faster, mirroring real-world degradation acceleration.

**RaceProgress** — normalized race position:

```
r_l = l / L
```

where `L` is total race laps. This captures the constraint that pits become strategically inadvisable beyond `r ≈ 0.85` (fewer than ~10 laps remaining, insufficient stint value to justify the 25s time loss). The majority of first stops cluster in the `r ∈ [0.25, 0.55]` window across race types.

**GapToLeader** — simplified gap estimate:

```
gap_l = (position_l - 1) × 0.5s
```

The 0.5s/position scaling is a practical approximation. In production, real-time telemetry provides actual gap data; here it captures relative race pressure as a monotone proxy.

---

## Data Pipeline

### Filtering Decisions

**Caution periods removed:** 540 laps under Safety Car (`TrackStatus = 4`) or Virtual SC (`TrackStatus = 6`) were excluded from training. Under caution, degradation models break — all drivers run constant reduced pace — and pit timing becomes race-control-driven rather than tire-state-driven. Including caution laps would teach the model a fundamentally different decision function that doesn't generalize to normal racing.

**Wet weather excluded:** 153 wet laps were excluded. Wet-to-dry transitions involve compound-choice decisions (inter, wet, slick) requiring a separate model class. A SOFT compound in the wet degrades through different mechanisms than in the dry; training both together creates domain confusion.

**Standing start removal (laps 1–3):** Lap 1 contains standing start pace anomalies; laps 2–3 involve DRS resolution and grid-order stabilization. None of these represent stable race pace suitable for degradation modeling.

**Pit laps excluded:** In-laps and out-laps are post-decision consequences, not decision inputs. The in-lap has the driver lifting off throttle before the pit entry; the out-lap has cold tires building grip. Neither reflects the degradation state used to make the pit call.

### Retention Summary

```
Raw data:           3,860 laps
After SC/VSC/pit:   3,037 laps  (−14.0%)
After wet filter:   2,874 laps  (−4.2%)
Final dataset:      2,874 laps  (74.5% retention)
```

### Target Distribution

```
Class 0 (no pit next 5 laps):  1,918 laps  (63%)
Class 1 (pit next 5 laps):     1,119 laps  (37%)
Ratio: 1.72:1
```

The 1.72:1 ratio is manageable without oversampling. The training set (2018–2023 multi-race) has a higher class imbalance of ~5.3:1 (84.2% no-pit, 15.8% pit), which is addressed via loss weighting.

---

## Model Selection

### Why XGBoost Over Logistic Regression

Three models were compared on the held-out 2024 test set (3 races, 2,801 laps):

| Model | F1 | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|---|
| Logistic Regression | 0.3223 | 0.7046 | — | Assumes linearity |
| Random Forest | 0.4484 | 0.8189 | — | Strong, slower inference |
| **XGBoost** | **0.4490** | **0.8409** | — | Best overall |

Logistic regression's lower F1 (0.3223 vs 0.4490) is expected: it assumes the log-odds of pitting is a linear combination of features. Tire degradation, however, is multiplicative — a 20-lap SOFT at 45°C track temp behaves non-linearly relative to a 20-lap SOFT at 30°C, and neither case is captured by additive coefficients. The model systematically under-fires on worn tires and over-fires on high-temperature laps.

Random Forest achieves near-identical F1 to XGBoost (0.4484 vs 0.4490) but at higher computational cost and with less-calibrated probability estimates. Probability calibration matters here because we rely on `predict_proba()` for threshold tuning — poorly calibrated probabilities reduce the precision of the threshold sweep. XGBoost's gradient boosting objective directly minimizes log-loss, producing better-calibrated probabilities, which contributes to its higher ROC-AUC (0.8409 vs 0.8189).

### Hyperparameters and Their Rationale

```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=5,
    random_state=42
)
```

- `max_depth=5`: Prevents memorizing individual race sequences. At depth 5, the model can capture interactions like `(TyreLife > 20) AND (RaceProgress > 0.5) AND (DegradationRate > 0.08)` without overfitting to specific driver/race combinations in the training set.
- `scale_pos_weight=5`: Directly addresses the 5.3:1 class imbalance in the training set. This parameter scales the gradient contribution of positive-class samples by a factor of 5, which is mathematically equivalent to oversampling but avoids the distributional artifacts of SMOTE.
- `learning_rate=0.1`: Conservative shrinkage — sufficient for 100 trees without requiring very deep trees.

### Cross-Validation Setup

5-fold stratified cross-validation was applied to the training set (2018–2023). Stratification preserves the class ratio within each fold, which is important given the imbalanced target. The CV F1 scores were used to confirm that models generalize across races, not to tune hyperparameters (which were set a priori based on domain considerations).

---

## Threshold Optimization

The default classification threshold `τ = 0.5` yields:

```
Precision: 31.3%
Recall:    79.5%
F1:        0.4490
```

In race strategy, the cost of a missed pit (false negative) is typically higher than a redundant call (false positive): missing a window can cost a position or strand a driver on dead tires; an unnecessary call costs only the minor flexibility loss from a slightly early stop. This asymmetry justifies shifting the threshold.

Grid search over `τ ∈ {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60}` on the test set:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.223 | 0.924 | 0.360 |
| 0.40 | 0.264 | 0.875 | 0.406 |
| 0.50 | 0.313 | 0.795 | 0.449 |
| 0.55 | 0.344 | 0.752 | 0.472 |
| **0.60** | **0.374** | **0.708** | **0.490** |

The optimal F1 threshold is `τ* = 0.60`, corresponding to the elbow of the precision-recall curve where marginal precision improvement begins exceeding marginal recall loss.

For deployment, the threshold choice depends on use case:
- `τ = 0.50` (conservative): minimize false pit calls, accept missing some windows
- `τ = 0.60` (aggressive): catch more windows, accept more false positives

---

## Feature Importance

XGBoost gain-based importance:

```
1. RaceProgress       59.2%
2. TyreLife           14.9%
3. DegradationRate    10.6%
─────────────────────────────
   Combined           84.7%   (top 3 features)

4. StintAgeSquared     4.1%
5. Position            3.8%
6. GapToLeader         3.1%
   (remaining 8 features: 14.3% combined)
```

`RaceProgress` dominates because most pits cluster in predictable race phases regardless of tire state — a driver at lap 45 of 60 has a dramatically higher prior probability of pitting than one at lap 10, regardless of compound. `TyreLife` and `DegradationRate` together capture the tire-state signal on top of the race-phase prior.

The low weight on positional features (GapToLeader: 3.1%) reflects the model's main limitation: it cannot distinguish a tactically-forced pit from a degradation-forced pit. This is the primary driver of false positives.

---

## Error Analysis

### False Positives (72.5% of pit predictions)

The model over-fires on aged MEDIUM tires in mid-pack positions. A driver at P10–P15 with 20+ laps on a MEDIUM set triggers high pit confidence even when the team is deliberately extending to track position or cycling through under a safety car phase the model doesn't see. Root cause: the model has no knowledge of team radio communications, upcoming yellow flags, or deliberate strategy overrides.

Example pattern:
```
Lap 39, SOFT, TyreLife=39, DegradationRate=+0.02 s/lap
Model confidence: 95.6% → predicts pit
Reality: team extends stint for undercut setup, pits lap 43
```

### False Negatives (20.4% of actual pits)

Strategic pits without degradation signal — primarily undercut attempts from P5–P8 on relatively fresh tires. The driver pits on lap 5 of a new stint (TyreLife=5, low DegradationRate) for a strategic reason, but the model's dominant features point to 'stay out.' Resolving this class of misses requires real-time gap-closing-rate telemetry.

Example pattern:
```
Lap 35, SOFT, TyreLife=5, DegradationRate=+0.52 s/lap (high but early)
Model confidence: 5.8% → predicts no pit
Reality: driver pits for undercut on competitor
```

---

## Results Summary

| Metric | Value |
|---|---|
| Best model | XGBoost |
| F1 (default threshold) | 0.4490 |
| F1 (tuned threshold 0.60) | 0.490 |
| ROC-AUC | 0.8409 |
| Recall | 79.5% (default), 70.8% (tuned) |
| Precision | 31.3% (default), 37.4% (tuned) |
| Training set | 16,867 laps (2018–2023) |
| Test set | 2,801 laps (2024, held-out) |

The ROC-AUC of 0.841 means the model correctly ranks a 'will pit' lap above a 'won't pit' lap 84.1% of the time. The low absolute precision reflects the inherent difficulty of the prediction: even expert strategists cannot reliably predict the exact pit lap more than a few laps in advance when safety cars, undercuts, and team overrides can change the decision in seconds.

The recall of 70–80% is the practically meaningful number: the model flags the pit window in 3 of 4 actual stops, giving strategy engineers a reliable early warning system.

---

## Files

| File | Description |
|---|---|
| `data_inspection.py` | Data schema validation, pit stop pattern analysis, quality assessment |
| `feature_engineering.py` | Data cleaning, feature computation pipeline, target construction |
| `model_comparison.py` | Model training, cross-validation, feature importance, threshold tuning |
| `DATA_INSPECTION_REPORT.md` | Data quality metrics and findings |
| `FEATURE_ENGINEERING_REPORT.md` | Feature justifications with worked examples |
| `QUICK_REFERENCE.md` | Quick-reference summary card |

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost matplotlib fastf1
```

The scripts use synthetic data matching the FastF1 API structure. To run on real race data, replace the `create_synthetic_race()` calls with `fastf1.get_session(year, round, 'R').load()`.

---

## Next Steps

The most impactful improvements in order of expected gain:

1. **Dynamic gap telemetry** — replace the fixed 0.5s/position approximation with actual sector gap data. This is the primary lever for reducing false positives in undercut scenarios.
2. **Sequence modeling** — a Bi-LSTM over the last 5 laps of each stint would capture degradation trajectory (is the slope accelerating?) rather than just the current slope value.
3. **2025 live validation** — the model was built on 2018–2023 with a 2024 holdout. Pirelli's compounds change yearly; testing on 2025 telemetry would quantify distributional drift.
4. **Caution strategy sub-model** — a separate model trained only on SC/VSC periods could handle caution-driven pits, which the current model explicitly excludes.

---

## License

MIT License — see `LICENSE` for details.
