# Phase 1: Core Infrastructure Setup Guide

**Goal**: Make the F1 Pit Strategy system production-ready with Docker containerization, secrets management, and structured logging.

**Timeline**: ~30 minutes to complete

**Status**: ✅ All files created and ready for deployment

---

## 📦 What's Included in Phase 1

### 1. **Containerization (Docker)**
- `Dockerfile` - Container image definition
- `docker-compose.yml` - Local development environment with Streamlit + PostgreSQL
- **Benefit**: Deploy to any cloud platform (AWS, GCP, Azure, Kubernetes)

### 2. **Configuration & Secrets Management**
- `.env.example` - Template for environment variables
- `config.py` - Settings management using Pydantic
- **Benefit**: No hard-coded credentials; support dev/staging/prod environments

### 3. **Structured Logging**
- `logging_config.py` - JSON logging configuration
- **Benefit**: Searchable logs in CloudWatch, Cloud Logging, Stackdriver

### 4. **Updated Files**
- `requirements.txt` - Pinned versions + new dependencies
- `sql_utils.py` - Updated to use config.py instead of hard-coded defaults

---

## 🚀 Quick Start (Local Development)

### Step 1: Prerequisites
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
# Or for Linux: sudo apt-get install docker.io docker-compose

docker --version  # Verify installation
docker-compose --version
```

### Step 2: Configure Environment
```bash
# Copy template to .env
cp .env.example .env

# Edit .env with your database credentials (for local dev, defaults are OK)
cat .env
```

### Step 3: Build & Run
```bash
# Build Docker image and start services
docker-compose up --build

# In another terminal, verify services are running
docker ps

# Access Streamlit dashboard
open http://localhost:8501

# Access PostgreSQL database
psql -h localhost -U postgres -d f1_pit_db
```

### Step 4: Verify Everything Works
```bash
# Check logs
docker-compose logs streamlit
docker-compose logs postgres

# Stop all services
docker-compose down

# Remove data (for clean restart)
docker-compose down -v
```

---

## 📋 Configuration Guide

### Environment Variables (.env file)

**Database Configuration**:
```env
DB_TYPE=postgresql          # postgresql, mysql, or sqlserver
DB_USER=postgres            # Database username
DB_PASSWORD=postgres        # Database password (change in production!)
DB_HOST=postgres            # Database hostname (postgres = docker service name)
DB_PORT=5432                # Database port
DB_NAME=f1_pit_db           # Database name
```

**Streamlit Configuration**:
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_LOGGER_LEVEL=info
```

**Logging Configuration**:
```env
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json             # json (production) or text (development)
```

**FastF1 Configuration**:
```env
FASTF1_API_TIMEOUT=10       # Timeout for FastF1 API calls
FASTF1_CACHE_DIR=.cache/    # Cache directory for FastF1 data
```

### Testing Configuration
```bash
# Check current configuration
python config.py
```

**Output**:
```
================================================================================
F1 PIT STRATEGY - CONFIGURATION
================================================================================

Environment: development
Log Level: INFO
Log Format: json

Database Configuration:
  Type: postgresql
  Host: postgres:5432
  Database: f1_pit_db
  User: postgres
  Password: ***

Streamlit Configuration:
  Address: 0.0.0.0:8501
  Logger Level: info

FastF1 Configuration:
  API Timeout: 10s
  Cache Dir: .cache/

================================================================================
```

---

## 🔍 How to Use in Your Code

### Using Config Settings

**Before** (Hard-coded):
```python
# sql_utils.py (OLD)
connector = SQLConnector(
    'postgresql',
    user='postgres',
    password='password',  # ❌ Hard-coded!
    host='localhost'
)
```

**After** (From Environment):
```python
# sql_utils.py (NEW)
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# Automatically loads from .env
connector = SQLConnector.from_settings(settings)
logger.info("✓ Database connected")
```

### Using Logging

**Before** (Print statements):
```python
print("Training model...")
print("Model F1-score: 0.432")
```

**After** (Structured logging):
```python
from logging_config import get_logger

logger = get_logger(__name__)

logger.info("Training model")
logger.info("Model trained", extra={"f1_score": 0.432, "roc_auc": 0.76})
```

**Output (JSON)**:
```json
{"timestamp": "2024-04-23T10:30:45.123456", "level": "INFO", "logger": "model_training", "message": "Model trained", "f1_score": 0.432, "roc_auc": 0.76}
```

---

## 📚 Database Setup (With Docker)

### Automatic (Recommended)
```bash
docker-compose up
# PostgreSQL starts automatically with empty database
```

### Manual Setup
```bash
# Connect to PostgreSQL
docker exec -it f1-pit-strategy-postgres psql -U postgres -d f1_pit_db

# Initialize database
python -c "
from sql_utils import Base, SQLConnector
from config import settings

connector = SQLConnector.from_settings(settings)
connector.init_db()  # Creates all tables
"
```

### Verify Tables
```bash
# Connect to database
docker exec -it f1-pit-strategy-postgres psql -U postgres -d f1_pit_db

# List tables
\dt

# Show races table structure
\d races

# Exit
\q
```

---

## 🔐 Security Checklist

- ✅ `.env` is in `.gitignore` (never committed)
- ✅ `.env.example` is public template (no passwords)
- ✅ Passwords come from environment, not code
- ✅ Supports cloud secrets managers (next phases)
- ✅ Logging doesn't output sensitive data by default

**For Production**:
1. Use cloud secrets (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
2. Rotate passwords regularly
3. Use strong passwords (20+ characters)
4. Enable SSL for database connections

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to database"
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# View logs
docker-compose logs postgres

# Verify connection string
python config.py
```

### Issue: "Port 8501 already in use"
```bash
# Find what's using port 8501
lsof -i :8501

# Kill the process
kill -9 <PID>

# Or use different port in .env
STREAMLIT_SERVER_PORT=8502
docker-compose up
```

### Issue: "config.py fails to load"
```bash
# Ensure .env exists
cp .env.example .env

# Verify format (no quotes needed)
cat .env | grep DB_

# Test config loading
python -c "from config import settings; print(settings)"
```

### Issue: "ModuleNotFoundError: No module named 'pydantic'"
```bash
# Reinstall dependencies with new versions
pip install -r requirements.txt

# Or in Docker:
docker-compose down
docker-compose up --build
```

---

## ✅ Phase 1 Verification Checklist

- [ ] Clone repo and navigate to directory
- [ ] `cp .env.example .env`
- [ ] `docker-compose up --build` completes without errors
- [ ] `docker ps` shows both `streamlit` and `postgres` containers running
- [ ] `http://localhost:8501` loads Streamlit dashboard
- [ ] `python config.py` shows correct configuration
- [ ] `docker-compose logs postgres` shows "database system is ready to accept connections"
- [ ] Can connect to database: `psql -h localhost -U postgres -d f1_pit_db`
- [ ] All tables exist: `\dt` shows races, laps, model_predictions, model_metrics
- [ ] Logs are in JSON format in `docker-compose logs streamlit`

---

## 📖 Next Steps

After Phase 1 is verified:

### Phase 2: Testing & Quality (1 week)
- Unit tests for models and API
- Code quality tools (black, flake8, mypy)
- Health check endpoints

### Phase 3: Cloud Deployment (1 week)
- Push to cloud registry (ECR, GCR, ACR)
- Deploy to Kubernetes or serverless platform
- Set up CI/CD pipeline

### Phase 4: Monitoring & Retraining (1 week)
- Prometheus metrics collection
- Cloud logging integration
- Automated model retraining pipeline

---

## 📞 Support

**Common issues and solutions**:
- Check logs: `docker-compose logs -f streamlit`
- Restart services: `docker-compose restart`
- Full rebuild: `docker-compose down -v && docker-compose up --build`
- View configuration: `python config.py`

---

## Summary

| Component | Status | Purpose |
|-----------|--------|---------|
| Dockerfile | ✅ Ready | Container image |
| docker-compose.yml | ✅ Ready | Local environment |
| .env.example | ✅ Ready | Configuration template |
| config.py | ✅ Ready | Settings management |
| logging_config.py | ✅ Ready | Structured logging |
| requirements.txt | ✅ Updated | Pinned versions |
| sql_utils.py | ✅ Updated | Uses config |

**You're now ready for local development!** 🚀

Next: Test locally, then move to Phase 2 (Testing & Quality).
