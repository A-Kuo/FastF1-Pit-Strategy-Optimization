# FastF1 Data Ingestion Pipeline

## Schema

### `races` table
Master table for F1 race sessions.

| Column | Type | Purpose |
|--------|------|---------|
| race_id | INT PRIMARY KEY | Unique race identifier |
| year | INT | F1 season year |
| race_name | VARCHAR(100) | Event name (e.g., "Bahrain") |
| circuit_name | VARCHAR(100) | Circuit name |
| session_date | TIMESTAMP | Race session date/time |
| num_drivers | INT | Number of drivers in race |
| num_laps | INT | Total race laps |
| created_at | TIMESTAMP | Ingestion timestamp |

### `laps` table
Individual lap telemetry with 29 columns.

#### Identifiers
| Column | Type | Purpose |
|--------|------|---------|
| lap_id | INT PRIMARY KEY | Unique lap identifier (auto-increment) |
| race_id | INT FOREIGN KEY | Reference to races table |
| session_key | VARCHAR(50) | FastF1 session key (for upsert) |
| driver_number | INT | Driver number (1–99) |
| driver_code | VARCHAR(3) | 3-letter driver code |
| team_name | VARCHAR(50) | Team name |
| lap_number | INT | Lap within race (1–N) |

#### Lap Performance
| Column | Type | Purpose |
|--------|------|---------|
| lap_time_seconds | FLOAT | Lap time in seconds |
| sector_1 | FLOAT | Sector 1 time (seconds) |
| sector_2 | FLOAT | Sector 2 time (seconds) |
| sector_3 | FLOAT | Sector 3 time (seconds) |
| is_pit_lap | BOOLEAN | Pit stop occurred this lap |
| pit_loss_seconds | FLOAT | Time lost during pit stop |

#### Tires & Stint
| Column | Type | Purpose |
|--------|------|---------|
| tyre_compound | VARCHAR(10) | SOFT / MEDIUM / HARD |
| tyre_life | INT | Laps on current tire (0–100+) |
| stint_number | INT | Pit stop number (1–3 typical) |

#### Position & Gaps
| Column | Type | Purpose |
|--------|------|---------|
| position | INT | Current position (1–20+) |
| position_text | VARCHAR(10) | Position as text |
| gap_to_leader | FLOAT | Gap to 1st place (seconds) |
| gap_to_car_in_front | FLOAT | Gap to car ahead (seconds) |

#### Weather & Track
| Column | Type | Purpose |
|--------|------|---------|
| air_temp | FLOAT | Ambient temperature (°C) |
| track_temp | FLOAT | Track surface temperature (°C) |
| wind_speed | FLOAT | Wind speed (km/h) |
| humidity | FLOAT | Relative humidity (%) |
| rainfall | BOOLEAN | Raining? |

#### Incidents & Safety
| Column | Type | Purpose |
|--------|------|---------|
| is_safety_car_lap | BOOLEAN | Safety car deployed |
| is_yellow_flag | BOOLEAN | Yellow flag |
| is_red_flag | BOOLEAN | Red flag |
| created_at | TIMESTAMP | Ingestion timestamp |

## Setup & Ingestion

### 1. Start PostgreSQL

```bash
docker-compose up -d postgres
```

Verify:
```bash
docker ps | grep postgres
psql -h localhost -U postgres -d f1_pit_db -c "SELECT 1;"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- fastf1 (3.8.3+) — FastF1 API library
- sqlalchemy (2.0.20+) — ORM
- psycopg2-binary (2.9.7+) — PostgreSQL adapter
- pandas, numpy, scikit-learn, xgboost

### 3. Ingest Data

**Full pipeline (2018–2024):**
```bash
python scripts/ingest.py
```

**Specific years:**
```bash
python scripts/ingest.py --years 2018,2019
```

**Specific races:**
```bash
python scripts/ingest.py --years 2024 --races 1,2,3
```

**Custom database URL:**
```bash
DATABASE_URL="postgresql://user:pass@host:5432/db" python scripts/ingest.py
```

### 4. Validate Data

```bash
python scripts/validate.py
```

Checks:
- Row counts per year
- Null rates per column
- Pit event rate (target: 1–15%)
- Numeric bounds (lap time, position, tyre life)
- Unique races and drivers

### 5. Engineer Features

```bash
python feature_engineering.py
```

Outputs:
- `models/X_train_scaled.npy` — Scaled training features (16,880 laps × 4 features)
- `models/X_test_scaled.npy` — Scaled test features (2,800 laps × 4 features)
- `models/y_train.npy` — Training target (pit next 5 laps)
- `models/y_test.npy` — Test target
- `models/scaler.pkl` — StandardScaler for inference

**Features engineered:**
1. **DegradationRate** — OLS slope of lap time vs. tyre life per stint
2. **StintAgeSquared** — tyre_life² (non-linear degradation proxy)
3. **RaceProgress** — lap / max_lap (normalized 0–1)
4. **PaceDelta** — driver lap time minus 5-lap rolling median

### 6. Train Model

```bash
python pipeline.py
```

Outputs:
- `models/xgboost_model.pkl` — Trained XGBoost model
- `models/metrics.pkl` — Evaluation metrics (ROC-AUC, F1, etc.)

## Environment Variables

```bash
# PostgreSQL (default: localhost)
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/f1_pit_db"

# FastF1 API cache directory
FASTF1_CACHE_DIR=".cache/"
```

## Troubleshooting

### "Connection refused" when ingesting
```bash
# Check postgres is running
docker ps | grep postgres

# Start it
docker-compose up -d postgres

# Wait 10s for health check
sleep 10
```

### "No data found for years" in feature_engineering.py
```bash
# Verify data was ingested
python scripts/validate.py

# Check row counts
docker exec f1-pit-strategy-postgres psql -U postgres -d f1_pit_db -c "SELECT COUNT(*) FROM laps;"
```

### "fastf1.exceptions.ConnectionError"
FastF1 requires internet access to download data from the official F1 API. Ensure your environment has network connectivity.

```bash
# Test FastF1 connectivity
python -c "import fastf1; session = fastf1.get_session(2024, 1, 'R'); session.load()"
```

## Data Quality Expectations

After full ingestion (2018–2024):

| Metric | Expected Range |
|--------|-----------------|
| Total laps | 100,000+ |
| Pit event rate | 10–15% |
| Null rate per feature | <5% |
| Lap time range | 60–140 seconds |
| Max position | 18–24 (DNFs) |
| Tyre life range | 0–100+ laps |

## References

- **FastF1 Documentation**: https://docs.fastf1.dev/
- **Data Schema**: See LapORM in `scripts/ingest.py`
- **Feature Engineering**: See `feature_engineering.py`
- **Pipeline**: See `pipeline.py` (model training)
