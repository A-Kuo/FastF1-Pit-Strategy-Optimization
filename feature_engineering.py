#!/usr/bin/env python3
"""
Feature Engineering from Real FastF1 Data
==========================================

Reads lap data from PostgreSQL, engineers 4 domain features:
  1. DegradationRate   - OLS slope of lap time vs. stint age
  2. StintAgeSquared   - Tyre age squared (non-linear proxy)
  3. RaceProgress      - lap / max_lap (normalized 0–1)
  4. PaceDelta         - Driver lap time minus 5-lap rolling median

Outputs:
  - models/X_train_scaled.npy, X_test_scaled.npy
  - models/y_train.npy, y_test.npy
  - models/scaler.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Allow running as `python feature_engineering.py` from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(levelname)8s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD DATA FROM POSTGRESQL
# ─────────────────────────────────────────────────────────────

def load_laps_from_db(db_url: str, years: list) -> pd.DataFrame:
    """Load lap data from PostgreSQL for specified years, including year column."""
    from scripts.ingest import RaceORM, LapORM

    engine = create_engine(db_url)

    with Session(engine) as session:
        from sqlalchemy import select
        from scripts.ingest import RaceORM as R, LapORM as L

        stmt = (
            select(L, R.year.label('year'))
            .join(R, L.race_id == R.race_id)
            .where(R.year.in_(years))
        )
        laps = pd.read_sql(stmt, engine)

    if laps.empty:
        raise ValueError(f"No data found for years {years} in database")

    logger.info(f"Loaded {len(laps):,} laps from {min(years)}–{max(years)}")
    return laps


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Engineer 4 domain features from raw lap data."""

    # 1. DegradationRate: OLS slope per (session_key, driver, stint)
    def compute_degradation(group):
        """OLS slope of lap_time vs. tyre_life."""
        if len(group) < 2 or group['tyre_life'].std() == 0:
            return 0.0
        X = group['tyre_life'].values.reshape(-1, 1)
        y = group['lap_time_seconds'].values
        try:
            slope = LinearRegression().fit(X, y).coef_[0]
            return float(np.clip(slope, -0.5, 0.5))  # Clip outliers
        except:
            return 0.0

    logger.info("Computing DegradationRate...")
    deg_rates = laps.groupby(['session_key', 'driver_number', 'stint_number']).apply(
        compute_degradation
    ).reset_index(name='DegradationRate')

    laps = laps.merge(
        deg_rates,
        on=['session_key', 'driver_number', 'stint_number'],
        how='left'
    )

    # 2. StintAgeSquared: tyre_life²
    logger.info("Computing StintAgeSquared...")
    laps['StintAgeSquared'] = laps['tyre_life'] ** 2

    # 3. RaceProgress: lap / max_lap per session
    logger.info("Computing RaceProgress...")
    max_laps = laps.groupby('session_key')['lap_number'].transform('max')
    laps['RaceProgress'] = laps['lap_number'] / max_laps

    # 4. PaceDelta: lap_time - rolling 5-lap median per driver per session
    logger.info("Computing PaceDelta...")
    laps = laps.sort_values(['session_key', 'driver_number', 'lap_number'])
    laps['RollingMedianTime'] = laps.groupby(['session_key', 'driver_number'])[
        'lap_time_seconds'
    ].transform(lambda x: x.rolling(5, min_periods=1).median())
    laps['PaceDelta'] = laps['lap_time_seconds'] - laps['RollingMedianTime']

    # 5. Target: pit_next_5_laps (engineered from is_pit_lap)
    logger.info("Computing target...")
    laps = laps.sort_values(['session_key', 'driver_number', 'lap_number'])
    laps['pit_next_5_laps'] = laps.groupby(['session_key', 'driver_number'])[
        'is_pit_lap'
    ].transform(lambda x: x.shift(-1).rolling(5, min_periods=1).max().fillna(0).astype(bool))

    # Fill NaN in new features
    laps['DegradationRate'] = laps['DegradationRate'].fillna(0.0)
    laps['PaceDelta'] = laps['PaceDelta'].fillna(0.0)

    return laps


# ─────────────────────────────────────────────────────────────
# SCALE & SPLIT
# ─────────────────────────────────────────────────────────────

def prepare_datasets(laps: pd.DataFrame, train_years: list, test_years: list):
    """Split into train/test and scale."""
    train_laps = laps[laps['year'].isin(train_years)]
    test_laps = laps[laps['year'].isin(test_years)]

    logger.info(f"Train: {len(train_laps):,} laps ({train_years[0]}–{train_years[-1]})")
    logger.info(f"Test:  {len(test_laps):,} laps ({test_years})")

    # Features
    feature_cols = ['DegradationRate', 'StintAgeSquared', 'RaceProgress', 'PaceDelta']

    X_train = train_laps[feature_cols].fillna(0).values
    y_train = train_laps['pit_next_5_laps'].astype(int).values
    X_test = test_laps[feature_cols].fillna(0).values
    y_test = test_laps['pit_next_5_laps'].astype(int).values

    # Scale
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"  Pit rate train: {y_train.mean()*100:.1f}%")
    logger.info(f"  Pit rate test:  {y_test.mean()*100:.1f}%")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    db_url = os.getenv('DATABASE_URL',
                      'postgresql://postgres:postgres@localhost:5432/f1_pit_db')

    # Load
    logger.info("="*60)
    logger.info("LOADING DATA FROM POSTGRESQL")
    logger.info("="*60)

    try:
        all_laps = load_laps_from_db(db_url, [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    except Exception as e:
        logger.error(f"Failed to load from DB: {e}")
        raise

    # Engineer
    logger.info("\n" + "="*60)
    logger.info("ENGINEERING FEATURES")
    logger.info("="*60)
    all_laps = engineer_features(all_laps)

    # Split & scale
    logger.info("\n" + "="*60)
    logger.info("SPLITTING & SCALING")
    logger.info("="*60)

    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_datasets(
        all_laps,
        train_years=[2018, 2019, 2020, 2021, 2022, 2023],
        test_years=[2024]
    )

    # Save
    logger.info("\n" + "="*60)
    logger.info("SAVING ARTIFACTS")
    logger.info("="*60)

    os.makedirs('models', exist_ok=True)
    np.save('models/X_train_scaled.npy', X_train)
    np.save('models/X_test_scaled.npy', X_test)
    np.save('models/y_train.npy', y_train)
    np.save('models/y_test.npy', y_test)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"✓ Saved X_train ({X_train.shape})")
    logger.info(f"✓ Saved X_test ({X_test.shape})")
    logger.info(f"✓ Saved y_train ({y_train.shape})")
    logger.info(f"✓ Saved y_test ({y_test.shape})")
    logger.info(f"✓ Saved scaler.pkl")
    logger.info(f"✓ Features: {feature_cols}")


if __name__ == '__main__':
    main()
