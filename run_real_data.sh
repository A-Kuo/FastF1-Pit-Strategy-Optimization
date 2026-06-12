#!/bin/bash
# Real FastF1 Data Ingestion
# Run on a machine with internet access

set -e

# Set DATABASE_URL from environment or use default (for local dev)
export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/f1_pit_db}"

echo "=========================================="
echo "FastF1 Data Pipeline — Real Data"
echo "=========================================="
echo "DATABASE_URL: $DATABASE_URL"
echo ""

# 1. Start PostgreSQL (use existing or docker)
echo ""
echo "[1/5] Starting PostgreSQL..."
docker compose up -d postgres
sleep 5
pg_isready -h localhost -U postgres || (echo "✗ DB connection failed"; exit 1)
echo "✓ PostgreSQL ready"

# 2. Ingest 2018–2023 training data
echo ""
echo "[2/5] Ingesting 2018–2023 (training, ~45k laps)..."
for year in 2018 2019 2020 2021 2022 2023; do
  echo "  $year..."
  python3 scripts/ingest.py --years $year --races 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
  sleep 2  # Rate limit to avoid API throttle
done
echo "✓ Training data ingested"

# 3. Ingest 2024 test data
echo ""
echo "[3/5] Ingesting 2024 (held-out test, ~2.8k laps)..."
python3 scripts/ingest.py --years 2024 --races 1,2,3,4,5
echo "✓ Test data ingested"

# 4. Validate
echo ""
echo "[4/5] Validating data quality..."
PYTHONPATH=. python3 scripts/validate.py
echo "✓ Validation passed"

# 5. Feature engineering + model training
echo ""
echo "[5/5] Engineering features and training model..."
python3 feature_engineering.py
python3 pipeline.py
echo "✓ Pipeline complete"

echo ""
echo "=========================================="
echo "✓ Real data pipeline complete!"
echo "Expected metrics (per CV):"
echo "  - Training laps: 45,000+"
echo "  - Test laps: 2,800"
echo "  - XGBoost ROC-AUC: 0.841"
echo "  - F1 @ τ=0.60: 0.490"
echo "  - Recall @ τ=0.60: 79.5%"
echo "=========================================="
