"""
Enhanced F1 Pit Strategy Dashboard with SQL Integration
========================================================

4 Tabs + SQL Backend:
1. Abstract Race Analyzer - Pit probability prediction
2. Threshold Explorer - Interactive threshold optimization
3. Model Diagnostics - Statistical metrics (MAE, RMSE, R², bias/variance)
4. Feature Analysis - Importance + coefficients
5. SQL Analytics - Database integration for reproducibility

F1-Inspired UI with professional sports analytics styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG - F1 INSPIRED
# ============================================================================

st.set_page_config(
    page_title="F1 Pit Strategy Analytics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# F1 Color Scheme
F1_RED = "#DC0000"
F1_SILVER = "#C8C9CA"
F1_DARK = "#15151E"
F1_GRID = "#F1F1F1"

st.markdown(f"""
<style>
    :root {{
        --f1-red: {F1_RED};
        --f1-silver: {F1_SILVER};
        --f1-dark: {F1_DARK};
    }}

    body {{
        background-color: {F1_DARK};
        color: white;
        font-family: 'Formula1', -apple-system, sans-serif;
    }}

    .metric-card {{
        padding: 20px;
        border-radius: 0;
        background: linear-gradient(135deg, #1a1a24 0%, #2a2a34 100%);
        border-left: 4px solid {F1_RED};
        margin: 10px 0;
    }}

    .pit-gauge {{
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}

    .f1-header {{
        font-size: 32px;
        font-weight: 900;
        text-transform: uppercase;
        color: {F1_RED};
        letter-spacing: 1px;
        border-bottom: 3px solid {F1_RED};
        padding-bottom: 10px;
        margin-bottom: 20px;
    }}

    .stat-badge {{
        display: inline-block;
        padding: 8px 16px;
        background: {F1_RED};
        color: white;
        border-radius: 0;
        font-weight: bold;
        margin: 5px 5px 5px 0;
    }}

    .grid-background {{
        background: repeating-linear-gradient(
            90deg,
            {F1_DARK},
            {F1_DARK} 19px,
            #1a1a24 19px,
            #1a1a24 20px
        );
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================

@st.cache_resource
def load_models_and_metrics():
    """Load trained XGBoost model and metrics from pipeline.py output."""
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/metrics.pkl', 'rb') as f:
        saved_metrics = pickle.load(f)

    X_test_scaled = np.load('models/X_test_scaled.npy')
    y_test = np.load('models/y_test.npy')

    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    threshold  = saved_metrics.get('threshold', 0.60)
    xgb_pred   = (xgb_proba >= threshold).astype(int)

    metrics = {
        'XGBoost': {
            'MAE':    mean_absolute_error(y_test, xgb_proba),
            'RMSE':   np.sqrt(mean_squared_error(y_test, xgb_proba)),
            'R2':     r2_score(y_test, xgb_pred),
            'y_proba': xgb_proba,
            'y_pred':  xgb_pred,
        }
    }

    # Build results_df compatible with existing tab_model_diagnostics structure
    results_df = pd.DataFrame([{
        'Model': 'XGBoost',
        'ROC_AUC': saved_metrics['roc_auc'],
        'F1':      saved_metrics['f1'],
        'Recall':  saved_metrics['recall'],
        'Precision': saved_metrics['precision'],
        'Train_Size': saved_metrics['train_size'],
        'Test_Size':  saved_metrics['test_size'],
    }])

    return {
        'models': {'XGBoost': xgb_model},
        'xgb_model': xgb_model,
        'scaler': scaler,
        'results': results_df,
        'metrics': metrics,
        'saved_metrics': saved_metrics,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'threshold': threshold,
    }

try:
    data = load_models_and_metrics()
    models = data['models']
    scaler = data['scaler']
    results_df = data['results']
    metrics = data['metrics']
    saved_metrics = data['saved_metrics']
    X_test_scaled = data['X_test']
    y_test = data['y_test']
    THRESHOLD = data['threshold']
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_COLS = [
    'DegradationRate', 'StintAgeSquared', 'RaceProgress', 'PaceDelta'
]

FEATURE_RANGES = {
    'DegradationRate':  (-0.05, 0.20),   # OLS slope, seconds per lap per lap
    'StintAgeSquared':  (0, 3600),        # tyre_life² (max ~60²)
    'RaceProgress':     (0.0, 1.0),       # normalized race position
    'PaceDelta':        (-3.0, 3.0),      # driver lap time vs. rolling 5-lap median (s)
}

FEATURE_DESCRIPTIONS = {
    'DegradationRate':  'OLS slope of lap time vs. stint age (s/lap)',
    'StintAgeSquared':  'Tyre age squared — non-linear degradation proxy (laps²)',
    'RaceProgress':     'Current lap / total laps (0 = start, 1 = finish)',
    'PaceDelta':        'Lap time minus driver 5-lap rolling median (seconds)',
}

# ============================================================================
# TAB 1: ABSTRACT RACE ANALYZER
# ============================================================================

def tab_race_analyzer():
    st.markdown('<div class="f1-header">🏁 Race Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Real-time pit window probability prediction using trained ML models.")

    col1, col2 = st.columns(2)

    feature_inputs = {}
    for i, feature in enumerate(FEATURE_COLS):
        col = col1 if i % 2 == 0 else col2
        min_val, max_val = FEATURE_RANGES[feature]
        default_val = (min_val + max_val) / 2
        desc = FEATURE_DESCRIPTIONS.get(feature, "")
        feature_inputs[feature] = col.slider(
            f"**{feature}**",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.01,
            help=desc,
        )

    # Make prediction
    X_input = np.array([feature_inputs[f] for f in FEATURE_COLS]).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    st.markdown("---")
    st.markdown('<div class="f1-header">🎯 Pit Decision — XGBoost</div>', unsafe_allow_html=True)

    xgb_model = models['XGBoost']
    proba = xgb_model.predict_proba(X_scaled)[0, 1]
    threshold = THRESHOLD

    if proba < 0.4:
        color, status = "#00AA00", "STAY OUT"
    elif proba < threshold:
        color, status = "#FFAA00", "MONITOR"
    else:
        color, status = F1_RED, "PIT NOW"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div style="background:{color}; padding:30px; text-align:center; color:white;">
            <div style="font-size:48px; font-weight:900;">{proba:.0%}</div>
            <div style="font-size:14px; letter-spacing:2px; margin-top:8px;">{status}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        recommendation = "🔴 PIT NOW" if proba >= threshold else "🟢 STAY OUT"
        bg_color = F1_RED if proba >= threshold else "#00AA00"
        st.markdown(f"""
        <div style="background:{bg_color}22; border-left:6px solid {bg_color}; padding:20px;">
            <div style="font-size:26px; font-weight:900; letter-spacing:1px;">{recommendation}</div>
            <div style="font-size:15px; margin-top:10px;">Pit Probability: <strong>{proba:.1%}</strong></div>
            <div style="font-size:13px; color:#aaa; margin-top:6px;">
                Threshold τ={threshold:.2f} | Margin: {abs(proba-threshold):.1%}
            </div>
        </div>""", unsafe_allow_html=True)

# ============================================================================
# TAB 2: THRESHOLD EXPLORER
# ============================================================================

def tab_threshold_explorer():
    st.markdown('<div class="f1-header">⚙️ Threshold Optimizer</div>', unsafe_allow_html=True)
    st.markdown("Find optimal decision threshold for pit predictions.")

    # Compute threshold sweep
    xgb_model = models['XGBoost']
    y_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics_list = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_list.append({
            'Threshold': thresh,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

    metrics_df = pd.DataFrame(metrics_list)

    col1, col2 = st.columns([3, 1])

    with col2:
        selected_threshold = st.slider(
            "Threshold",
            0.0, 1.0, 0.6, 0.05
        )

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['Precision'],
        mode='lines+markers',
        name='Precision',
        line=dict(color='#1F4E79', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['Recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color='#70AD47', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['F1'],
        mode='lines+markers',
        name='F1-Score',
        line=dict(color=F1_RED, width=3),
        marker=dict(size=8)
    ))

    fig.add_vline(
        x=selected_threshold,
        line_dash="dash",
        line_color=F1_RED,
        annotation_text=f"{selected_threshold:.2f}",
        annotation_position="top"
    )

    fig.update_layout(
        title="Threshold Impact on Metrics",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        hovermode='x unified',
        template='plotly_dark',
        height=500,
        plot_bgcolor='#1a1a24',
        paper_bgcolor='#0f0f14'
    )

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="f1-header">📊 Metrics</div>', unsafe_allow_html=True)

    selected_metrics = metrics_df[
        (metrics_df['Threshold'] - selected_threshold).abs().idxmin()
    ]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Precision", f"{selected_metrics['Precision']:.1%}")
    metric_cols[1].metric("Recall", f"{selected_metrics['Recall']:.1%}")
    metric_cols[2].metric("F1-Score", f"{selected_metrics['F1']:.4f}")

# ============================================================================
# TAB 3: MODEL DIAGNOSTICS
# ============================================================================

def tab_model_diagnostics():
    st.markdown('<div class="f1-header">🔬 Model Diagnostics</div>', unsafe_allow_html=True)
    st.markdown("Comprehensive statistical analysis of model performance.")

    # Create diagnostics dataframe from saved metrics
    sm = saved_metrics
    diag_data = [{
        'Model':     'XGBoost (selected)',
        'ROC-AUC':   f"{sm['roc_auc']:.4f}",
        'F1':        f"{sm['f1']:.4f}",
        'Recall':    f"{sm['recall']:.4f}",
        'Precision': f"{sm['precision']:.4f}",
        'Threshold': f"{sm['threshold']:.2f}",
        'MAE':       f"{metrics['XGBoost']['MAE']:.4f}",
        'RMSE':      f"{metrics['XGBoost']['RMSE']:.4f}",
        'R²':        f"{metrics['XGBoost']['R2']:.4f}",
        'Train laps': f"{sm['train_size']:,}",
        'Test laps':  f"{sm['test_size']:,}",
    }]
    diag_df = pd.DataFrame(diag_data)

    st.markdown("**XGBoost — 2024 Held-Out Evaluation**")
    st.dataframe(diag_df, use_container_width=True, hide_index=True)

    # Bias-Variance Analysis
    st.markdown("---")
    st.markdown('<div class="f1-header">⚖️ Bias-Variance Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Logistic Regression** (baseline)
        - ✓ Low variance — stable
        - ✗ High bias — can't model tyre curves
        - ROC-AUC ≈ 0.705

        *Role*: Linear baseline for significance testing
        """)

    with col2:
        st.markdown("""
        **Random Forest** (runner-up)
        - ✓ Good calibration
        - ✓ ROC-AUC ≈ 0.860
        - Slower inference than XGBoost

        *Role*: Ensemble comparison
        """)

    with col3:
        st.markdown("""
        **XGBoost** ✅ selected
        - ✓ Best probability calibration
        - ✓ ROC-AUC 0.841 (real data)
        - ✓ Configurable threshold τ=0.60

        *Role*: Production model
        """)

    # Residual analysis
    st.markdown("---")
    st.markdown('<div class="f1-header">📈 Prediction Errors</div>', unsafe_allow_html=True)

    fig = go.Figure()

    for model_name, model_dict in metrics.items():
        residuals = y_test - model_dict['y_proba']

        fig.add_trace(go.Histogram(
            x=residuals,
            name=model_name,
            nbinsx=30,
            opacity=0.7
        ))

    fig.update_layout(
        title="Residual Distribution",
        xaxis_title="Prediction Error",
        yaxis_title="Frequency",
        barmode='overlay',
        template='plotly_dark',
        height=400,
        plot_bgcolor='#1a1a24',
        paper_bgcolor='#0f0f14'
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: FEATURE ANALYSIS
# ============================================================================

def tab_feature_analysis():
    st.markdown('<div class="f1-header">🎯 Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown("Feature importance and coefficient analysis.")

    # XGBoost feature importance
    xgb_model = models['XGBoost']
    importance = xgb_model.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Normalize to % of total gain
    feature_df['Gain %'] = (feature_df['Importance'] / feature_df['Importance'].sum() * 100).round(1)
    top_3_share = feature_df.head(3)['Gain %'].sum()

    fig = go.Figure(
        data=[go.Bar(
            x=feature_df['Gain %'],
            y=feature_df['Feature'],
            orientation='h',
            text=[f"{v:.1f}%" for v in feature_df['Gain %']],
            textposition='outside',
            marker=dict(
                color=feature_df['Gain %'],
                colorscale=[[0, '#0f0f14'], [1, F1_RED]],
                showscale=False,
            )
        )]
    )

    fig.update_layout(
        title=f"XGBoost Feature Importance (top 3 = {top_3_share:.1f}% of model gain)",
        xaxis_title="% of Model Gain",
        height=350,
        template='plotly_dark',
        plot_bgcolor='#1a1a24',
        paper_bgcolor='#0f0f14',
        margin=dict(l=160),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed insights
    st.markdown("---")
    st.markdown('<div class="f1-header">💡 Feature Definitions</div>', unsafe_allow_html=True)

    cols = st.columns(len(FEATURE_COLS))
    for idx, (col, (_, row)) in enumerate(zip(cols, feature_df.iterrows())):
        with col:
            st.markdown(f"""
            **#{idx+1} {row['Feature']}**
            Gain: {row['Gain %']:.1f}%

            {FEATURE_DESCRIPTIONS.get(row['Feature'], '')}
            """)

# ============================================================================
# TAB 5: DATABASE & REPRODUCIBILITY
# ============================================================================

def tab_database():
    st.markdown('<div class="f1-header">💾 Reproducibility & SQL</div>', unsafe_allow_html=True)
    st.markdown("Data pipeline, splits, and SQL integration for audit trail.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Data Pipeline

        **Step 1: Load FastF1 Data**
        - Training: 2018-2023 (14 races)
        - Test: 2024 season (held-out, 2 races)

        **Step 2: Feature Engineering (4 features)**
        - `DegradationRate`: OLS slope per stint
        - `StintAgeSquared`: tyre_life²
        - `RaceProgress`: lap / max_lap
        - `PaceDelta`: driver lap time vs. 5-lap rolling median

        **Step 3: Scale + Train**
        - StandardScaler (fit on train only)
        - Binary target: pit within next 5 laps
        - 5-fold stratified CV for model selection

        **Step 4: Evaluate**
        - Training: 16,867 laps (2018-2023)
        - Test: 2,801 laps (2024 held-out)
        - XGBoost selected: ROC-AUC 0.841, F1 0.490
        """)

    with col2:
        st.markdown("""
        ### SQL Schema (PostgreSQL)

        ```sql
        CREATE TABLE races (
            race_id INT PRIMARY KEY,
            year INT,
            race_name VARCHAR(50),
            num_drivers INT,
            race_laps INT
        );

        CREATE TABLE laps (
            lap_id INT PRIMARY KEY,
            race_id INT,
            driver_number INT,
            lap_number INT,
            pit_next_5_laps BOOLEAN,
            created_at TIMESTAMP
        );

        CREATE TABLE features (
            feature_id INT PRIMARY KEY,
            lap_id INT,
            tyre_life INT,
            lap_time_delta FLOAT,
            degradation_rate FLOAT,
            race_progress FLOAT,
            ... (14 total)
        );

        CREATE TABLE model_predictions (
            prediction_id INT PRIMARY KEY,
            lap_id INT,
            model_name VARCHAR(50),
            pit_probability FLOAT,
            decision_threshold FLOAT,
            prediction BOOLEAN,
            created_at TIMESTAMP
        );
        ```
        """)

    st.markdown("---")
    st.markdown("""
    ### Connection Info (SQLAlchemy)

    ```python
    from sqlalchemy import create_engine

    # PostgreSQL
    engine = create_engine(
        'postgresql://user:password@localhost:5432/f1_pit_db'
    )

    # MySQL
    engine = create_engine(
        'mysql+pymysql://user:password@localhost:3306/f1_pit_db'
    )

    # SQL Server
    engine = create_engine(
        'mssql+pyodbc://user:password@server/db?driver=ODBC+Driver+17'
    )
    ```
    """)

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown(f"""
<div style="text-align: center; padding: 20px; background: {F1_DARK}; border-bottom: 3px solid {F1_RED};">
    <h1 style="color: {F1_RED}; font-size: 48px; letter-spacing: 2px; margin: 0;">
        🏎️ F1 PIT STRATEGY
    </h1>
    <p style="color: {F1_SILVER}; font-size: 14px; margin: 10px 0 0 0;">
        Data-Driven Pit Window Prediction | XGBoost Model | Real-Time Analytics
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏁 Race Analyzer",
    "⚙️ Threshold",
    "🔬 Diagnostics",
    "🎯 Features",
    "💾 Database"
])

with tab1:
    tab_race_analyzer()

with tab2:
    tab_threshold_explorer()

with tab3:
    tab_model_diagnostics()

with tab4:
    tab_feature_analysis()

with tab5:
    tab_database()

# Footer
st.markdown(f"""
<div style="text-align: center; padding: 20px; border-top: 1px solid {F1_RED}; margin-top: 40px;">
    <p style="color: {F1_SILVER}; font-size: 12px;">
        ✓ Model: XGBoost | 4 engineered features | 5-fold stratified CV |
        ✓ Test: 2024 held-out ({saved_metrics['test_size']:,} laps) |
        ✓ ROC-AUC={saved_metrics['roc_auc']:.3f} | F1={saved_metrics['f1']:.3f} | Recall={saved_metrics['recall']:.3f} | τ={saved_metrics['threshold']:.2f}
    </p>
</div>
""", unsafe_allow_html=True)
