#!/usr/bin/env python3
"""
FastF1 Data Ingestion → PostgreSQL
==================================

Fetches F1 race sessions (2018–2023 training, 2024 test) from FastF1 API,
normalizes lap data, and upserts into PostgreSQL via SQLAlchemy.

Usage:
    python scripts/ingest.py [--years 2018,2019] [--races 1,2,3]
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from sqlalchemy.dialects.postgresql import insert

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# SCHEMA (mirrors sql_utils.py)
# ─────────────────────────────────────────────────────────────

Base = declarative_base()

class RaceORM(Base):
    """F1 race metadata"""
    __tablename__ = 'races'

    race_id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    race_name = Column(String(100), nullable=False)
    circuit_name = Column(String(100))
    session_date = Column(DateTime)
    num_drivers = Column(Integer)
    num_laps = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Race {self.year} {self.race_name}>"


class LapORM(Base):
    """Individual lap with engineered features (from FastF1)"""
    __tablename__ = 'laps'

    lap_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey('races.race_id'), nullable=False)
    session_key = Column(String(50), nullable=False)  # FastF1 session_key (upsert key)
    driver_number = Column(Integer, nullable=False)
    driver_code = Column(String(3))
    team_name = Column(String(50))
    lap_number = Column(Integer, nullable=False)

    # Lap time & performance
    lap_time_seconds = Column(Float)  # in seconds
    sector_1 = Column(Float)
    sector_2 = Column(Float)
    sector_3 = Column(Float)
    is_pit_lap = Column(Boolean, default=False)
    pit_loss_seconds = Column(Float)

    # Tire & stint
    tyre_compound = Column(String(10))  # SOFT, MEDIUM, HARD
    tyre_life = Column(Integer)  # laps on current tire
    stint_number = Column(Integer)

    # Position & gaps
    position = Column(Integer)
    position_text = Column(String(10))
    gap_to_leader = Column(Float)
    gap_to_car_in_front = Column(Float)

    # Weather & track
    air_temp = Column(Float)
    track_temp = Column(Float)
    wind_speed = Column(Float)
    humidity = Column(Float)
    rainfall = Column(Boolean)

    # Safety car / incident flags
    is_safety_car_lap = Column(Boolean, default=False)
    is_yellow_flag = Column(Boolean, default=False)
    is_red_flag = Column(Boolean, default=False)

    # Created timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Lap {self.session_key} D{self.driver_number} L{self.lap_number}>"


# ─────────────────────────────────────────────────────────────
# FASTF1 INGESTION
# ─────────────────────────────────────────────────────────────

def fetch_session(year: int, race_num: int) -> tuple[pd.DataFrame, dict]:
    """
    Fetch a FastF1 session and return normalized lap data + race metadata.

    Returns:
        (laps_df, race_dict) or (None, None) if fetch fails
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('.cache/')

        logger.info(f"Fetching {year} race {race_num}...")
        session = fastf1.get_session(year, race_num, 'R')
        session.load(weather=True, telemetry=False)

        laps = session.laps
        if laps.empty:
            logger.warning(f"  No laps found for {year} R{race_num}")
            return None, None

        # Normalize columns
        laps_normalized = normalize_lap_dataframe(laps, session)

        race_meta = {
            'year': year,
            'race_name': session.event['EventName'],
            'circuit_name': session.event['Circuit'],
            'session_date': session.date if hasattr(session, 'date') else None,
            'num_drivers': laps['Driver'].nunique(),
            'num_laps': laps['LapNumber'].max(),
        }

        logger.info(f"  ✓ {year} R{race_num}: {len(laps_normalized)} laps, "
                   f"{race_meta['num_drivers']} drivers")
        return laps_normalized, race_meta

    except Exception as e:
        logger.error(f"  ✗ {year} R{race_num}: {e}")
        return None, None


def normalize_lap_dataframe(laps: pd.DataFrame, session) -> pd.DataFrame:
    """Normalize FastF1 lap DataFrame to match LapORM schema."""
    df = laps.copy()

    # Add session key for upsert
    session_key = f"{session.event['Season']}-{session.event['RoundNumber']}"
    df['session_key'] = session_key

    # Standardize column names / fill missing
    column_map = {
        'Time': 'lap_time_seconds',
        'Driver': 'driver_number',
        'LapNumber': 'lap_number',
        'Compound': 'tyre_compound',
        'TyreLife': 'tyre_life',
        'Stint': 'stint_number',
        'Position': 'position',
        'PositionText': 'position_text',
        'Sector1Time': 'sector_1',
        'Sector2Time': 'sector_2',
        'Sector3Time': 'sector_3',
        'PitInTime': 'pit_in_time',
        'PitOutTime': 'pit_out_time',
        'AirTemp': 'air_temp',
        'TrackTemp': 'track_temp',
        'WindSpeed': 'wind_speed',
        'Humidity': 'humidity',
        'Rainfall': 'rainfall',
    }

    # Rename available columns
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Convert lap_time_seconds to float (in seconds)
    if 'lap_time_seconds' in df.columns:
        df['lap_time_seconds'] = pd.to_timedelta(df['lap_time_seconds']).dt.total_seconds()

    # Pit lap detection (has PitOutTime or PitInTime)
    df['is_pit_lap'] = df[['pit_in_time', 'pit_out_time']].notna().any(axis=1)

    # Safety car detection (if available in session)
    df['is_safety_car_lap'] = False  # Would need session event data
    df['is_yellow_flag'] = False
    df['is_red_flag'] = False

    # Pit loss (rough estimate: difference between normal/pit lap time)
    df['pit_loss_seconds'] = np.nan

    # Fill NaN in numeric columns with 0
    numeric_cols = ['lap_time_seconds', 'sector_1', 'sector_2', 'sector_3',
                    'gap_to_leader', 'gap_to_car_in_front', 'air_temp', 'track_temp',
                    'wind_speed', 'humidity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Extract driver code if available
    if 'Driver' in df.columns and hasattr(df.iloc[0]['Driver'], 'code'):
        df['driver_code'] = df['Driver'].apply(lambda x: x.code if hasattr(x, 'code') else None)

    # Add team name
    if 'Team' in df.columns:
        df['team_name'] = df['Team'].apply(lambda x: str(x) if x else None)

    # Select final columns
    final_cols = [
        'session_key', 'driver_number', 'driver_code', 'team_name',
        'lap_number', 'lap_time_seconds', 'sector_1', 'sector_2', 'sector_3',
        'is_pit_lap', 'pit_loss_seconds', 'tyre_compound', 'tyre_life', 'stint_number',
        'position', 'position_text', 'gap_to_leader', 'gap_to_car_in_front',
        'air_temp', 'track_temp', 'wind_speed', 'humidity', 'rainfall',
        'is_safety_car_lap', 'is_yellow_flag', 'is_red_flag'
    ]

    df = df[[col for col in final_cols if col in df.columns]]

    return df


def upsert_laps(engine, laps_df: pd.DataFrame, race_id: int):
    """
    Upsert lap data into PostgreSQL.
    Upsert key: (session_key, driver_number, lap_number)
    """
    if laps_df.empty:
        return

    laps_df['race_id'] = race_id

    with Session(engine) as session:
        # PostgreSQL UPSERT using INSERT ... ON CONFLICT DO UPDATE
        stmt = insert(LapORM).values(laps_df.to_dict('records'))
        stmt = stmt.on_conflict_do_update(
            index_elements=['session_key', 'driver_number', 'lap_number'],
            set_={col.name: col for col in stmt.excluded}
        )
        session.execute(stmt)
        session.commit()
        logger.info(f"  ✓ Upserted {len(laps_df)} laps for race_id={race_id}")


def main():
    parser = argparse.ArgumentParser(description='Ingest FastF1 data into PostgreSQL')
    parser.add_argument('--years', default='2018,2019,2020,2021,2022,2023,2024',
                       help='Years to ingest (comma-separated)')
    parser.add_argument('--races', default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24',
                       help='Race numbers to try per year (comma-separated)')
    parser.add_argument('--db-url', default=None,
                       help='Database URL (or use DATABASE_URL env var)')
    args = parser.parse_args()

    # Database connection
    db_url = args.db_url or os.getenv('DATABASE_URL',
                                      'postgresql://postgres:postgres@localhost:5432/f1_pit_db')
    logger.info(f"Connecting to {db_url.split('@')[1]}")
    engine = create_engine(db_url)

    # Create tables
    Base.metadata.create_all(engine)
    logger.info("✓ Schema created/verified")

    # Parse years and races
    years = list(map(int, args.years.split(',')))
    races = list(map(int, args.races.split(',')))

    # Ingest data
    stats = {'races': 0, 'laps': 0, 'errors': 0}

    for year in years:
        for race_num in races:
            laps_df, race_meta = fetch_session(year, race_num)

            if laps_df is None:
                stats['errors'] += 1
                continue

            # Insert race metadata
            with Session(engine) as session:
                existing = session.query(RaceORM).filter_by(
                    year=race_meta['year'],
                    race_name=race_meta['race_name']
                ).first()

                if not existing:
                    race = RaceORM(**race_meta)
                    session.add(race)
                    session.commit()
                    race_id = race.race_id
                else:
                    race_id = existing.race_id

            # Upsert laps
            upsert_laps(engine, laps_df, race_id)
            stats['races'] += 1
            stats['laps'] += len(laps_df)

    # Summary
    logger.info("\n" + "="*60)
    logger.info(f"Ingestion complete:")
    logger.info(f"  Races ingested: {stats['races']}")
    logger.info(f"  Laps ingested:  {stats['laps']:,}")
    logger.info(f"  Errors:         {stats['errors']}")
    logger.info("="*60)

    # Print row counts by year
    with Session(engine) as session:
        for year in sorted(set(years)):
            count = session.query(LapORM).join(RaceORM).filter(
                RaceORM.year == year
            ).count()
            if count > 0:
                logger.info(f"  {year}: {count:,} laps")


if __name__ == '__main__':
    main()
