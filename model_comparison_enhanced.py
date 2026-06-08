"""
TASK 2: Enhanced Model Comparison with PR-AUC
==============================================

Train Logistic Regression, Random Forest, XGBoost on real structured data.
Evaluate with Precision, Recall, F1, ROC-AUC, PR-AUC.
Generate precision-recall curve comparison plot.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, auc
)
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("TASK 2: ENHANCED MODEL COMPARISON WITH PR-AUC")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading cleaned datasets...")
train_clean = pd.read_parquet('data/train_clean.parquet')
test_clean = pd.read_parquet('data/test_clean.parquet')

feature_cols = [
    'TyreLife', 'LapTimeDelta', 'DegradationRate', 'StintAgeSquared',
    'RaceProgress', 'Position', 'GapToLeader', 'GapToCarInFront',
    'PitDeltaEstimated', 'StopsCompleted', 'StopsRemaining', 'PitStrategyID',
    'AirTemp', 'TrackTemp'
]

X_train = train_clean[feature_cols].fillna(0)
y_train = train_clean['pit_next_5_laps']
X_test = test_clean[feature_cols].fillna(0)
y_test = test_clean['pit_next_5_laps']

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training: {len(X_train)} laps")
print(f"✓ Test: {len(X_test)} laps")
print(f"✓ Features: {len(feature_cols)}")

# ============================================================================
# TASK 2: TRAIN & EVALUATE MODELS
# ============================================================================

print("\nTraining models...")

models = {}
results = []
pr_curves = {}

# Model 1: Logistic Regression
print("\n1. Logistic Regression (baseline)...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr

y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
pr_auc_lr = average_precision_score(y_test, y_proba_lr)

results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': prec_lr,
    'Recall': rec_lr,
    'F1': f1_lr,
    'ROC-AUC': roc_auc_lr,
    'PR-AUC': pr_auc_lr
})

# Store PR curve data
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_proba_lr)
pr_curves['Logistic Regression'] = (recall_lr, precision_lr)

print(f"   F1={f1_lr:.4f}, ROC-AUC={roc_auc_lr:.4f}, PR-AUC={pr_auc_lr:.4f}")

# Model 2: Random Forest
print("\n2. Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
pr_auc_rf = average_precision_score(y_test, y_proba_rf)

results.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': prec_rf,
    'Recall': rec_rf,
    'F1': f1_rf,
    'ROC-AUC': roc_auc_rf,
    'PR-AUC': pr_auc_rf
})

# Store PR curve data
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
pr_curves['Random Forest'] = (recall_rf, precision_rf)

print(f"   F1={f1_rf:.4f}, ROC-AUC={roc_auc_rf:.4f}, PR-AUC={pr_auc_rf:.4f}")

# Model 3: XGBoost
print("\n3. XGBoost...")
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=5, random_state=42, verbosity=0)
xgb.fit(X_train_scaled, y_train)
models['XGBoost'] = xgb

y_pred_xgb = xgb.predict(X_test_scaled)
y_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

prec_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
rec_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
pr_auc_xgb = average_precision_score(y_test, y_proba_xgb)

results.append({
    'Model': 'XGBoost',
    'Accuracy': accuracy_score(y_test, y_pred_xgb),
    'Precision': prec_xgb,
    'Recall': rec_xgb,
    'F1': f1_xgb,
    'ROC-AUC': roc_auc_xgb,
    'PR-AUC': pr_auc_xgb
})

# Store PR curve data
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
pr_curves['XGBoost'] = (recall_xgb, precision_xgb)

print(f"   F1={f1_xgb:.4f}, ROC-AUC={roc_auc_xgb:.4f}, PR-AUC={pr_auc_xgb:.4f}")

# ============================================================================
# RESULTS TABLE
# ============================================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE (With PR-AUC)")
print("=" * 80)
print("\n" + results_df.to_string(index=False))

# Compute improvements
best_f1_idx = results_df['F1'].idxmax()
best_model = results_df.loc[best_f1_idx, 'Model']
best_f1 = results_df.loc[best_f1_idx, 'F1']
baseline_f1 = results_df[results_df['Model'] == 'Logistic Regression']['F1'].values[0]

print(f"\n→ Best model: {best_model}")
print(f"  F1-Score improvement: {best_f1 - baseline_f1:+.4f} ({(best_f1/baseline_f1 - 1)*100:+.1f}%)")
print(f"  PR-AUC: {results_df.loc[best_f1_idx, 'PR-AUC']:.4f} (area under precision-recall curve)")

# Save results
results_df.to_csv('results/model_comparison.csv', index=False)
print(f"\n✓ Results saved to results/model_comparison.csv")

# ============================================================================
# PRECISION-RECALL CURVE PLOT
# ============================================================================

print("\nGenerating precision-recall curve comparison...")

fig = go.Figure()

colors = {
    'Logistic Regression': '#1F4E79',
    'Random Forest': '#70AD47',
    'XGBoost': '#C5504D'
}

for model_name, (recall, precision) in pr_curves.items():
    auc_score = results_df[results_df['Model'] == model_name]['PR-AUC'].values[0]

    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f"{model_name} (AUC={auc_score:.3f})",
        line=dict(color=colors[model_name], width=3)
    ))

fig.update_layout(
    title="Precision-Recall Curves: Model Comparison",
    xaxis_title="Recall (True Positive Rate)",
    yaxis_title="Precision (Positive Predictive Value)",
    hovermode='x unified',
    template='plotly_white',
    width=900,
    height=600,
    font=dict(size=12)
)

fig.write_html('results/pr_curve_comparison.html')
print("✓ Interactive plot saved to results/pr_curve_comparison.html")

# Also save static PNG version
try:
    fig.write_image('results/pr_curve_comparison.png', width=900, height=600)
    print("✓ Static PNG saved to results/pr_curve_comparison.png")
except:
    print("⚠ PNG export requires kaleido; using HTML instead")

# Save models
for model_name, model in models.items():
    with open(f'models/{model_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)

print(f"\n✓ Models saved to models/")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✓ TASK 2 COMPLETE")
print("=" * 80)

print(f"""
PR-AUC Values (Precision-Recall Area Under Curve):
  • Logistic Regression: {pr_auc_lr:.4f}
  • Random Forest:       {pr_auc_rf:.4f}
  • XGBoost:             {pr_auc_xgb:.4f}

Interpretation:
  PR-AUC measures model's ability to correctly identify pit opportunities
  while minimizing false alarms. Higher = better for imbalanced datasets
  (15% pit rate in this case).

  XGBoost: {pr_auc_xgb:.4f} → When model predicts pit, precision ~{prec_xgb*100:.0f}%
           Recall {rec_xgb:.2%} → Catches ~{rec_xgb*100:.0f}% of actual pit windows

Best for Production: {best_model}
  • F1-Score: {best_f1:.4f}
  • ROC-AUC: {results_df.loc[best_f1_idx, 'ROC-AUC']:.4f}
  • PR-AUC: {results_df.loc[best_f1_idx, 'PR-AUC']:.4f}

Files Generated:
  ✓ results/model_comparison.csv (metrics table)
  ✓ results/pr_curve_comparison.html (interactive plot)
  ✓ models/*.pkl (trained models)
""")
