#!/usr/bin/env python3
"""
Data Validation
===============

Validates ingested FastF1 data:
- Row counts per year
- Null rates per column
- Pit event rate (1–15% of laps)
- Numeric column bounds
"""

import os
import logging
from sqlalchemy import create_engine, func, and_
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format='%(levelname)8s | %(message)s')
logger = logging.getLogger(__name__)

# Import schema
from scripts.ingest import RaceORM, LapORM, Base


def validate_data():
    """Run all validations."""
    db_url = os.getenv('DATABASE_URL',
                      'postgresql://postgres:postgres@localhost:5432/f1_pit_db')

    logger.info(f"Connecting to {db_url.split('@')[1]}")
    engine = create_engine(db_url)

    with Session(engine) as session:
        # 1. Row counts by year
        logger.info("\n" + "="*60)
        logger.info("ROW COUNTS BY YEAR")
        logger.info("="*60)

        for year in sorted([2018, 2019, 2020, 2021, 2022, 2023, 2024]):
            count = session.query(func.count(LapORM.lap_id)).join(RaceORM).filter(
                RaceORM.year == year
            ).scalar() or 0
            logger.info(f"  {year}: {count:,} laps")

        # 2. Null rates
        logger.info("\n" + "="*60)
        logger.info("NULL RATES (should be low)")
        logger.info("="*60)

        total_laps = session.query(func.count(LapORM.lap_id)).scalar() or 1

        columns_to_check = [
            'lap_time_seconds', 'tyre_compound', 'tyre_life',
            'position', 'gap_to_leader', 'air_temp', 'track_temp'
        ]

        for col in columns_to_check:
            null_count = session.query(func.count(LapORM.lap_id)).filter(
                getattr(LapORM, col).is_(None)
            ).scalar() or 0
            null_pct = (null_count / total_laps * 100) if total_laps > 0 else 0
            status = "✓" if null_pct < 5 else "⚠"
            logger.info(f"  {status} {col:20s}: {null_pct:5.1f}% null")

        # 3. Pit event rate
        logger.info("\n" + "="*60)
        logger.info("PIT EVENT RATE (target: 1–15%)")
        logger.info("="*60)

        pit_count = session.query(func.count(LapORM.lap_id)).filter(
            LapORM.is_pit_lap == True
        ).scalar() or 0
        pit_rate = (pit_count / total_laps * 100) if total_laps > 0 else 0

        if 1 <= pit_rate <= 15:
            status = "✓"
        elif pit_rate == 0:
            status = "✗ (missing pit data)"
        else:
            status = "⚠"

        logger.info(f"  {status} Pit laps: {pit_count:,} / {total_laps:,} ({pit_rate:.1f}%)")

        # 4. Numeric bounds
        logger.info("\n" + "="*60)
        logger.info("NUMERIC BOUNDS")
        logger.info("="*60)

        # Lap time should be 60–120 seconds
        min_time = session.query(func.min(LapORM.lap_time_seconds)).scalar()
        max_time = session.query(func.max(LapORM.lap_time_seconds)).scalar()
        logger.info(f"  Lap time: {min_time:.1f}s – {max_time:.1f}s (expect 60–120s)")

        # Position should be 1–20+
        min_pos = session.query(func.min(LapORM.position)).filter(
            LapORM.position > 0
        ).scalar()
        max_pos = session.query(func.max(LapORM.position)).scalar()
        logger.info(f"  Position: {min_pos}–{max_pos} (expect 1–24)")

        # Tyre life should be 0–100
        min_life = session.query(func.min(LapORM.tyre_life)).scalar()
        max_life = session.query(func.max(LapORM.tyre_life)).scalar()
        logger.info(f"  Tyre life: {min_life}–{max_life} laps (expect 0–100)")

        # 5. Unique drivers/races
        logger.info("\n" + "="*60)
        logger.info("UNIQUE ENTITIES")
        logger.info("="*60)

        num_races = session.query(func.count(RaceORM.race_id)).scalar() or 0
        num_drivers = session.query(func.count(func.distinct(LapORM.driver_number))).scalar() or 0

        logger.info(f"  {num_races:3d} races")
        logger.info(f"  {num_drivers:3d} unique drivers")

        logger.info("\n" + "="*60)
        logger.info("✓ Validation complete")
        logger.info("="*60)


if __name__ == '__main__':
    validate_data()
