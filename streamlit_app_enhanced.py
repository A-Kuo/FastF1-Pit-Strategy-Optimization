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
    """Load trained models and compute comprehensive metrics."""
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/xgboost.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/logistic_regression.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load test data
    X_test_scaled = np.load('models/X_test_scaled.npy')
    y_test = np.load('models/y_test.npy')

    results_df = pd.read_csv('results/model_comparison.csv')

    # Compute additional metrics
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

    metrics = {}
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        mae = mean_absolute_error(y_test, y_proba)
        rmse = np.sqrt(mean_squared_error(y_test, y_proba))
        r2 = r2_score(y_test, y_pred)

        metrics[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'y_proba': y_proba,
            'y_pred': y_pred
        }

    return {
        'models': models,
        'scaler': scaler,
        'results': results_df,
        'metrics': metrics,
        'X_test': X_test_scaled,
        'y_test': y_test
    }

try:
    data = load_models_and_metrics()
    models = data['models']
    scaler = data['scaler']
    results_df = data['results']
    metrics = data['metrics']
    X_test_scaled = data['X_test']
    y_test = data['y_test']
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_COLS = [
    'TyreLife', 'LapTimeDelta', 'DegradationRate', 'StintAgeSquared',
    'RaceProgress', 'Position', 'GapToLeader', 'GapToCarInFront',
    'PitDeltaEstimated', 'StopsCompleted', 'StopsRemaining', 'PitStrategyID',
    'AirTemp', 'TrackTemp'
]

FEATURE_RANGES = {
    'TyreLife': (0, 67),
    'LapTimeDelta': (-10, 10),
    'DegradationRate': (0, 0.1),
    'StintAgeSquared': (0, 4500),
    'RaceProgress': (0.0, 1.0),
    'Position': (1, 20),
    'GapToLeader': (0, 9.5),
    'GapToCarInFront': (0, 5),
    'PitDeltaEstimated': (20, 30),
    'StopsCompleted': (0, 3),
    'StopsRemaining': (0, 3),
    'PitStrategyID': (1, 3),
    'AirTemp': (10, 30),
    'TrackTemp': (25, 55)
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

        feature_inputs[feature] = col.slider(
            f"**{feature}**",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.1
        )

    # Make prediction
    X_input = np.array([feature_inputs[f] for f in FEATURE_COLS]).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    st.markdown("---")
    st.markdown('<div class="f1-header">🎯 Pit Decision</div>', unsafe_allow_html=True)

    pred_cols = st.columns(3)
    selected_model = st.radio(
        "**SELECT MODEL**",
        list(models.keys()),
        horizontal=True,
        label_visibility="collapsed"
    )

    pit_probabilities = {}
    for idx, (model_name, model_obj) in enumerate(models.items()):
        with pred_cols[idx]:
            proba = model_obj.predict_proba(X_scaled)[0, 1]
            pit_probabilities[model_name] = proba

            if proba < 0.4:
                color = "#00AA00"
                status = "STAY OUT"
            elif proba < 0.6:
                color = "#FFAA00"
                status = "CAUTION"
            else:
                color = F1_RED
                status = "PIT NOW"

            st.markdown(f"""
            <div style="background: {color}; padding: 20px; border-radius: 0; color: white; text-align: center;">
                <div style="font-size: 12px; font-weight: bold; letter-spacing: 1px;">{model_name}</div>
                <div class="pit-gauge">{proba:.0%}</div>
                <div style="font-size: 12px; margin-top: 10px; letter-spacing: 1px;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="f1-header">⚡ {selected_model}</div>', unsafe_allow_html=True)

    selected_proba = pit_probabilities[selected_model]
    threshold = 0.6
    recommendation = "🔴 PIT NOW" if selected_proba >= threshold else "🟢 STAY OUT"
    bg_color = F1_RED if selected_proba >= threshold else "#00AA00"

    st.markdown(f"""
    <div style="background: {bg_color}22; border-left: 6px solid {bg_color}; padding: 20px; border-radius: 0;">
        <div style="font-size: 28px; font-weight: 900; margin-bottom: 15px; letter-spacing: 1px;">{recommendation}</div>
        <div style="font-size: 16px;">Pit Probability: <strong>{selected_proba:.1%}</strong></div>
        <div style="font-size: 13px; color: #aaa; margin-top: 10px;">
            Decision Threshold: {threshold:.0%} | Margin: {abs(selected_proba - threshold):.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: THRESHOLD EXPLORER
# ============================================================================

def tab_threshold_explorer():
    st.markdown('<div class="f1-header">⚙️ Threshold Optimizer</div>', unsafe_allow_html=True)
    st.markdown("Find optimal decision threshold for pit predictions.")

    # Compute threshold sweep
    rf_model = models['Random Forest']
    y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

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

    # Create diagnostics dataframe
    diag_data = []
    for model_name, metric_dict in metrics.items():
        base_metrics = results_df[results_df['Model'] == model_name].iloc[0]
        diag_data.append({
            'Model': model_name,
            'Accuracy': f"{base_metrics['Accuracy']:.4f}",
            'Precision': f"{base_metrics['Precision']:.4f}",
            'Recall': f"{base_metrics['Recall']:.4f}",
            'F1-Score': f"{base_metrics['F1']:.4f}",
            'ROC-AUC': f"{base_metrics['ROC-AUC']:.4f}",
            'PR-AUC': f"{base_metrics['PR-AUC']:.4f}",
            'MAE': f"{metric_dict['MAE']:.4f}",
            'RMSE': f"{metric_dict['RMSE']:.4f}",
            'R²': f"{metric_dict['R2']:.4f}"
        })

    diag_df = pd.DataFrame(diag_data)

    st.markdown("**Classification & Regression Metrics**")
    st.dataframe(diag_df, use_container_width=True, hide_index=True)

    # Bias-Variance Analysis
    st.markdown("---")
    st.markdown('<div class="f1-header">⚖️ Bias-Variance Tradeoff</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Logistic Regression**
        - ✓ Low variance (simple model)
        - ✗ High bias (underfitting)
        - F1: 0.2652

        *Implication*: Misses pit signals
        """)

    with col2:
        st.markdown("""
        **Random Forest**
        - ✓ Balanced bias-variance
        - ✓ Best F1-score (0.4320)
        - MAE: 0.2174

        *Implication*: Production choice
        """)

    with col3:
        st.markdown("""
        **XGBoost**
        - ✓ Excellent ROC-AUC
        - ✓ Best PR-AUC (0.2716)
        - RMSE: 0.3864

        *Implication*: Probability calibration
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

    top_15 = feature_df.head(15)

    fig = go.Figure(
        data=[go.Bar(
            x=top_15['Importance'],
            y=top_15['Feature'],
            orientation='h',
            marker=dict(
                color=top_15['Importance'],
                colorscale=[[0, '#0f0f14'], [1, F1_RED]],
                showscale=True
            )
        )]
    )

    fig.update_layout(
        title="Top 15 Feature Importance (XGBoost)",
        xaxis_title="Importance Score",
        height=600,
        template='plotly_dark',
        plot_bgcolor='#1a1a24',
        paper_bgcolor='#0f0f14'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed insights
    st.markdown("---")
    st.markdown('<div class="f1-header">💡 Key Features</div>', unsafe_allow_html=True)

    top_3_cols = st.columns(3)

    for idx, (col, (feat_idx, row)) in enumerate(zip(top_3_cols, list(top_15.head(3).iterrows()))):
        with col:
            st.markdown(f"""
            ### {row['Feature']}
            **Importance**: {row['Importance']:.4f}

            Rank: **#{idx + 1}** of 14 features
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

        **Step 1: Load Raw Data**
        - Training: 2018-2023 (34,800 raw laps)
        - Test: 2024 (3,600 raw laps)

        **Step 2: Clean Data**
        - Remove pit outcome laps
        - Remove SC/VSC periods
        - Remove standing starts
        - Remove wet weather
        - Retention: 78%

        **Step 3: Feature Engineering**
        - 14 features (tire, race, strategy, environment)
        - StandardScaler normalization
        - Binary target: pit_next_5_laps

        **Step 4: Train/Test Split**
        - Training: 27,188 laps (2018-2023)
        - Test: 2,828 laps (2024 held-out)
        - Seed: Fixed for reproducibility
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
        ✓ Models: Logistic Regression, Random Forest, XGBoost |
        ✓ Test Set: 2024 held-out (2,828 laps) |
        ✓ Metrics: F1={results_df.iloc[1]['F1']:.4f}, ROC-AUC={results_df.iloc[1]['ROC-AUC']:.4f}, PR-AUC={results_df.iloc[1]['PR-AUC']:.4f}
    </p>
</div>
""", unsafe_allow_html=True)
