"""
TASK 3: Interactive Streamlit Dashboard for F1 Pit Strategy
============================================================

4 Tabs:
1. Abstract Race Analyzer - Input features, get pit probability
2. Threshold Explorer - Interactive threshold tuning
3. Feature Importance - XGBoost feature importance
4. Model Performance - Model comparison metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="F1 Pit Strategy Analyzer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card { padding: 20px; border-radius: 10px; background: #f0f2f6; margin: 10px 0; }
    .pit-probability { font-size: 32px; font-weight: bold; text-align: center; }
    .recommendation { font-size: 20px; font-weight: bold; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================

@st.cache_resource
def load_models_and_data():
    """Load trained models and reference data."""
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    with open('models/xgboost.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    with open('models/logistic_regression.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    results_df = pd.read_csv('results/model_comparison.csv')

    return {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Logistic Regression': lr_model,
        'scaler': scaler,
        'results': results_df
    }

try:
    models_data = load_models_and_data()
    models = {k: v for k, v in models_data.items() if k != 'scaler' and k != 'results'}
    scaler = models_data['scaler']
    results_df = models_data['results']
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

FEATURE_HELP = {
    'TyreLife': 'Cumulative laps on current tire (0-67)',
    'LapTimeDelta': 'Pace vs driver median (±10s)',
    'DegradationRate': 'Tire degradation slope (s/lap)',
    'StintAgeSquared': '(TyreLife)² captures accelerating degradation',
    'RaceProgress': 'Fraction of race completed (0-1)',
    'Position': 'Driver position on track (1-20)',
    'GapToLeader': 'Estimated seconds behind leader',
    'GapToCarInFront': 'Position delta to car ahead',
    'PitDeltaEstimated': 'Pit stop time loss (s)',
    'StopsCompleted': 'Number of pits executed',
    'StopsRemaining': 'Estimated pits left',
    'PitStrategyID': 'Strategy type (1-stop, 2-stop, etc)',
    'AirTemp': 'Ambient temperature (°C)',
    'TrackTemp': 'Track surface temperature (°C)'
}

# ============================================================================
# TAB 1: ABSTRACT RACE ANALYZER
# ============================================================================

def tab_race_analyzer():
    st.header("🏁 Abstract Race Analyzer")
    st.markdown("Input race conditions and tire state to get pit probability prediction.")

    col1, col2 = st.columns(2)

    feature_inputs = {}

    for i, feature in enumerate(FEATURE_COLS):
        col = col1 if i % 2 == 0 else col2
        min_val, max_val = FEATURE_RANGES[feature]
        default_val = (min_val + max_val) / 2

        feature_inputs[feature] = col.slider(
            feature,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.1,
            help=FEATURE_HELP[feature]
        )

    # Make prediction
    X_input = np.array([feature_inputs[f] for f in FEATURE_COLS]).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    # Get predictions from all models
    st.markdown("---")
    st.subheader("🎯 Model Predictions")

    pred_cols = st.columns(3)

    selected_model = st.radio("Select model for recommendation:", list(models.keys()), horizontal=True)
    model = models[selected_model]

    pit_probabilities = {}
    predictions = {}

    for idx, (model_name, model_obj) in enumerate(models.items()):
        with pred_cols[idx]:
            if model_name == 'Logistic Regression':
                proba = model_obj.predict_proba(X_scaled)[0, 1]
            else:
                proba = model_obj.predict_proba(X_scaled)[0, 1]

            pit_probabilities[model_name] = proba

            # Color gauge
            if proba < 0.4:
                color = "green"
                status = "NO PIT"
            elif proba < 0.6:
                color = "orange"
                status = "CAUTION"
            else:
                color = "red"
                status = "PIT"

            st.markdown(f"""
            <div style="background: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 14px; font-weight: bold;">{model_name}</div>
                <div style="font-size: 32px; font-weight: bold;">{proba:.1%}</div>
                <div style="font-size: 12px; margin-top: 10px;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

    # Selected model recommendation
    st.markdown("---")
    st.subheader(f"📊 {selected_model} Recommendation")

    selected_proba = pit_probabilities[selected_model]
    threshold = 0.6
    recommendation = "🔴 PIT NOW" if selected_proba >= threshold else "🟢 STAY OUT"
    bg_color = "#fee" if selected_proba >= threshold else "#efe"
    border_color = "red" if selected_proba >= threshold else "green"

    st.markdown(f"""
    <div style="background: {bg_color}; padding: 20px; border-radius: 10px; border-left: 4px solid {border_color};">
        <div style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">{recommendation}</div>
        <div style="font-size: 16px;">Pit Probability: <strong>{selected_proba:.1%}</strong></div>
        <div style="font-size: 14px; color: #666; margin-top: 10px;">
            Threshold: {threshold:.0%} | Confidence: {abs(selected_proba - threshold):.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance for this input
    st.markdown("---")
    st.subheader("📈 Feature Values Overview")

    feature_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Value': [feature_inputs[f] for f in FEATURE_COLS],
        'Range': [f"{FEATURE_RANGES[f][0]:.1f} - {FEATURE_RANGES[f][1]:.1f}" for f in FEATURE_COLS]
    })

    st.dataframe(feature_df, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 2: THRESHOLD EXPLORER
# ============================================================================

def tab_threshold_explorer():
    st.header("⚙️ Threshold Explorer")
    st.markdown("Adjust decision threshold and see impact on Precision, Recall, and F1-Score.")

    # Load test data and predictions for threshold sweep
    X_test_scaled = np.load('models/X_test_scaled.npy')
    y_test = np.load('models/y_test.npy')

    # Get probabilities from best model (Random Forest)
    rf_model = models['Random Forest']
    y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Compute threshold sweep
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics.append({
            'Threshold': thresh,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

    metrics_df = pd.DataFrame(metrics)

    # Interactive threshold selector
    col1, col2 = st.columns([3, 1])

    with col2:
        selected_threshold = st.slider(
            "Decision Threshold",
            0.0, 1.0, 0.6, 0.05,
            help="Adjust the pit probability cutoff"
        )

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['Precision'],
        mode='lines+markers',
        name='Precision',
        line=dict(color='#1F4E79', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['Recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color='#70AD47', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['F1'],
        mode='lines+markers',
        name='F1-Score',
        line=dict(color='#C5504D', width=3)
    ))

    # Add vertical line for selected threshold
    fig.add_vline(
        x=selected_threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Selected: {selected_threshold:.2f}",
        annotation_position="top"
    )

    fig.update_layout(
        title="Threshold Impact on Model Metrics",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # Metrics at selected threshold
    st.markdown("---")
    st.subheader(f"📊 Metrics at Threshold {selected_threshold:.2f}")

    selected_metrics = metrics_df[metrics_df['Threshold'] == metrics_df['Threshold'].round(2).round(2).eq(round(selected_threshold, 2))]

    if len(selected_metrics) > 0:
        row = selected_metrics.iloc[0]
        metric_cols = st.columns(3)

        metric_cols[0].metric("Precision", f"{row['Precision']:.1%}", help="Of predicted pits, how many are correct?")
        metric_cols[1].metric("Recall", f"{row['Recall']:.1%}", help="Of actual pits, how many are caught?")
        metric_cols[2].metric("F1-Score", f"{row['F1']:.4f}", help="Harmonic mean of precision and recall")

    # Recommendations table
    st.markdown("---")
    st.subheader("💡 Threshold Recommendations")

    recommendations = pd.DataFrame([
        {"Use Case": "Conservative (minimize false pits)", "Threshold": 0.60, "Rationale": "High precision, fewer unnecessary calls"},
        {"Use Case": "Balanced", "Threshold": 0.55, "Rationale": "Medium precision & recall"},
        {"Use Case": "Aggressive (catch pits)", "Threshold": 0.50, "Rationale": "High recall, accept false positives"},
    ])

    st.dataframe(recommendations, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3: FEATURE IMPORTANCE
# ============================================================================

def tab_feature_importance():
    st.header("🎯 Feature Importance Analysis")
    st.markdown("Understanding which features drive pit stop predictions in XGBoost.")

    # Get feature importance from XGBoost
    xgb_model = models['XGBoost']
    importance = xgb_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Top 15 features
    top_15 = feature_importance_df.head(15)

    fig = go.Figure(
        data=[go.Bar(
            x=top_15['Importance'],
            y=top_15['Feature'],
            orientation='h',
            marker=dict(
                color=top_15['Importance'],
                colorscale='Viridis',
                showscale=True
            )
        )]
    )

    fig.update_layout(
        title="Top 15 Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed explanations for top 3
    st.markdown("---")
    st.subheader("📚 Top 3 Features Explained")

    top_3_explanations = {
        'TyreLife': {
            'importance': feature_importance_df[feature_importance_df['Feature'] == 'TyreLife']['Importance'].values[0],
            'explanation': """
            **Tire Life** is the cumulative number of laps on the current tire set. As tire life increases:
            - Tire degradation accelerates exponentially
            - Grip decreases (lap times increase)
            - Pit window closes (must pit before critical loss of grip)

            **Impact**: High tire life (>40 laps) is a strong signal to pit soon.
            """
        },
        'DegradationRate': {
            'importance': feature_importance_df[feature_importance_df['Feature'] == 'DegradationRate']['Importance'].values[0],
            'explanation': """
            **Degradation Rate** is the slope of lap time vs. tire life (s/lap). It captures how fast tires are losing grip:
            - High degradation (>0.05 s/lap) = soft compound, must pit sooner
            - Low degradation (0.01 s/lap) = hard compound, can extend stint

            **Impact**: Combined with tire life, degradation predicts remaining stint length.
            """
        },
        'RaceProgress': {
            'importance': feature_importance_df[feature_importance_df['Feature'] == 'RaceProgress']['Importance'].values[0],
            'explanation': """
            **Race Progress** (0-1) indicates how far through the race we are:
            - Early race (0.0-0.3): More pit stops remaining, strategic decisions dominate
            - Mid-race (0.3-0.7): Peak physical degradation, tires fail fastest
            - Late race (0.7-1.0): Final stint, pit window closing

            **Impact**: Race phase drives overall pit strategy and timing.
            """
        }
    }

    for i, (feature, info) in enumerate(list(top_3_explanations.items())[:3], 1):
        st.markdown(f"""
        ### {i}. {feature}
        **Importance Score**: {info['importance']:.4f}

        {info['explanation']}
        """)

# ============================================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================================

def tab_model_performance():
    st.header("📊 Model Performance Comparison")
    st.markdown("Comprehensive evaluation of all trained models.")

    # Results table
    st.subheader("Model Metrics")
    results_display = results_df.copy()
    results_display = results_display.round(4)

    st.dataframe(results_display, use_container_width=True, hide_index=True)

    # Comparison charts
    st.markdown("---")
    st.subheader("Visual Comparison")

    col1, col2 = st.columns(2)

    # F1 & ROC-AUC
    with col1:
        fig1 = go.Figure(data=[
            go.Bar(name='F1-Score', x=results_df['Model'], y=results_df['F1'], marker_color='#1F4E79'),
            go.Bar(name='ROC-AUC', x=results_df['Model'], y=results_df['ROC-AUC'], marker_color='#70AD47')
        ])
        fig1.update_layout(
            title="F1-Score vs ROC-AUC",
            barmode='group',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Precision & Recall
    with col2:
        fig2 = go.Figure(data=[
            go.Bar(name='Precision', x=results_df['Model'], y=results_df['Precision'], marker_color='#C5504D'),
            go.Bar(name='Recall', x=results_df['Model'], y=results_df['Recall'], marker_color='#FFC000')
        ])
        fig2.update_layout(
            title="Precision vs Recall",
            barmode='group',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # PR-AUC
    col1, col2, col3 = st.columns(3)

    with col1:
        fig3 = go.Figure(data=[
            go.Bar(x=results_df['Model'], y=results_df['PR-AUC'], marker_color='#70AD47')
        ])
        fig3.update_layout(
            title="PR-AUC Scores",
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("---")
        st.subheader("📌 Key Insights")
        st.markdown("""
        - **Random Forest** achieves best F1-score (0.4320) and balanced performance
        - **XGBoost** has highest PR-AUC (0.2716), best for imbalanced data
        - **Recall >90%** means catching most pit opportunities
        - **Low Precision** (~28%) indicates room for feature engineering
        """)

    with col3:
        st.markdown("---")
        st.subheader("🎯 Recommendation")
        st.markdown("""
        **Use: Random Forest for Production**

        ✅ Best overall F1-score
        ✅ Good ROC-AUC (0.76)
        ✅ High recall (91.6%)
        ✅ Fast inference

        **Threshold: 0.60**
        Conservative pit calls with good coverage.
        """)

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🏎️ Formula 1 Pit Strategy Analyzer")
st.markdown("Data-driven pit stop timing prediction using machine learning")

# Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🏁 Race Analyzer",
    "⚙️ Threshold Explorer",
    "🎯 Feature Importance",
    "📊 Model Performance"
])

with tab1:
    tab_race_analyzer()

with tab2:
    tab_threshold_explorer()

with tab3:
    tab_feature_importance()

with tab4:
    tab_model_performance()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    F1 Pit Strategy Optimization | Built with Streamlit | Models: Logistic Regression, Random Forest, XGBoost
</div>
""", unsafe_allow_html=True)
