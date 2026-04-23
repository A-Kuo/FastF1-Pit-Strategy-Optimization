"""
SQL Integration for F1 Pit Strategy Analytics
==============================================

Provides SQLAlchemy ORM models and utility functions for:
- PostgreSQL, MySQL, SQL Server support
- Model predictions audit trail
- Reproducibility and data lineage
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime
import pandas as pd
from typing import Optional

Base = declarative_base()

# ============================================================================
# ORM MODELS
# ============================================================================

class RaceORM(Base):
    """F1 race metadata"""
    __tablename__ = 'races'

    race_id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    race_name = Column(String(50), nullable=False)
    num_drivers = Column(Integer, default=20)
    race_laps = Column(Integer, default=60)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Race {self.year} {self.race_name}>"


class LapORM(Base):
    """Individual lap with features"""
    __tablename__ = 'laps'

    lap_id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.race_id'), nullable=False)
    driver_number = Column(Integer, nullable=False)
    lap_number = Column(Integer, nullable=False)

    # Target
    pit_next_5_laps = Column(Boolean, nullable=False)

    # Features: Tire Degradation
    tyre_life = Column(Integer, nullable=False)
    lap_time_delta = Column(Float, nullable=False)
    degradation_rate = Column(Float, nullable=False)
    stint_age_squared = Column(Integer, nullable=False)

    # Features: Race State
    race_progress = Column(Float, nullable=False)
    position = Column(Integer, nullable=False)
    gap_to_leader = Column(Float, nullable=False)
    gap_to_car_in_front = Column(Float, nullable=False)

    # Features: Strategy
    pit_delta_estimated = Column(Float, nullable=False)
    stops_completed = Column(Integer, nullable=False)
    stops_remaining = Column(Integer, nullable=False)
    pit_strategy_id = Column(Integer, nullable=False)

    # Features: Environment
    air_temp = Column(Float, nullable=False)
    track_temp = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Lap {self.race_id} Driver{self.driver_number} Lap{self.lap_number}>"


class ModelPredictionORM(Base):
    """Model predictions audit trail"""
    __tablename__ = 'model_predictions'

    prediction_id = Column(Integer, primary_key=True)
    lap_id = Column(Integer, ForeignKey('laps.lap_id'), nullable=False)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20), default='1.0')

    # Prediction
    pit_probability = Column(Float, nullable=False)
    decision_threshold = Column(Float, nullable=False)
    pit_prediction = Column(Boolean, nullable=False)

    # Ground truth
    actual_pit = Column(Boolean, nullable=True)
    is_correct = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.model_name} Prob={self.pit_probability:.3f}>"


class ModelMetricsORM(Base):
    """Model evaluation metrics"""
    __tablename__ = 'model_metrics'

    metrics_id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    test_set_name = Column(String(50), default='test_2024')

    # Classification metrics
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)

    # Regression metrics
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    r2 = Column(Float, nullable=False)

    # Threshold info
    decision_threshold = Column(Float, default=0.6)
    threshold_type = Column(String(20), default='conservative')

    # Metadata
    num_trees = Column(Integer, nullable=True)
    max_depth = Column(Integer, nullable=True)
    learning_rate = Column(Float, nullable=True)

    evaluated_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Metrics {self.model_name} F1={self.f1_score:.4f}>"


# ============================================================================
# CONNECTION UTILITIES
# ============================================================================

class SQLConnector:
    """Database connection manager with support for multiple backends"""

    def __init__(self, db_type: str, **kwargs):
        """
        Initialize database connection.

        Args:
            db_type: 'postgresql', 'mysql', or 'sqlserver'
            **kwargs: Connection parameters (user, password, host, database, etc.)
        """
        self.db_type = db_type
        self.engine = self._create_engine(db_type, **kwargs)

    def _create_engine(self, db_type: str, **kwargs):
        """Create SQLAlchemy engine for specified database type"""

        if db_type == 'postgresql':
            user = kwargs.get('user', 'postgres')
            password = kwargs.get('password', '')
            host = kwargs.get('host', 'localhost')
            port = kwargs.get('port', 5432)
            database = kwargs.get('database', 'f1_pit_db')
            url = f'postgresql://{user}:{password}@{host}:{port}/{database}'

        elif db_type == 'mysql':
            user = kwargs.get('user', 'root')
            password = kwargs.get('password', '')
            host = kwargs.get('host', 'localhost')
            port = kwargs.get('port', 3306)
            database = kwargs.get('database', 'f1_pit_db')
            url = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'

        elif db_type == 'sqlserver':
            user = kwargs.get('user', 'sa')
            password = kwargs.get('password', '')
            server = kwargs.get('server', 'localhost')
            database = kwargs.get('database', 'f1_pit_db')
            driver = kwargs.get('driver', 'ODBC Driver 17 for SQL Server')
            url = f'mssql+pyodbc://{user}:{password}@{server}/{database}?driver={driver}'

        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        return create_engine(url, echo=False)

    def init_db(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        print(f"✓ Database initialized ({self.db_type})")

    def get_session(self):
        """Get new database session"""
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(bind=self.engine)
        return SessionLocal()

    def insert_laps(self, df: pd.DataFrame, race_id: int):
        """Insert lap data from DataFrame"""
        session = self.get_session()
        try:
            for _, row in df.iterrows():
                lap = LapORM(
                    race_id=race_id,
                    driver_number=int(row['DriverNumber']),
                    lap_number=int(row['LapNumber']),
                    pit_next_5_laps=bool(row['pit_next_5_laps']),
                    tyre_life=int(row['TyreLife']),
                    lap_time_delta=float(row['LapTimeDelta']),
                    degradation_rate=float(row['DegradationRate']),
                    stint_age_squared=int(row['StintAgeSquared']),
                    race_progress=float(row['RaceProgress']),
                    position=int(row['Position']),
                    gap_to_leader=float(row['GapToLeader']),
                    gap_to_car_in_front=float(row['GapToCarInFront']),
                    pit_delta_estimated=float(row['PitDeltaEstimated']),
                    stops_completed=int(row['StopsCompleted']),
                    stops_remaining=int(row['StopsRemaining']),
                    pit_strategy_id=int(row['PitStrategyID']),
                    air_temp=float(row['AirTemp']),
                    track_temp=float(row['TrackTemp'])
                )
                session.add(lap)
            session.commit()
            print(f"✓ Inserted {len(df)} laps for race {race_id}")
        finally:
            session.close()

    def insert_predictions(self, predictions_df: pd.DataFrame):
        """Insert model predictions"""
        session = self.get_session()
        try:
            for _, row in predictions_df.iterrows():
                pred = ModelPredictionORM(
                    lap_id=int(row['lap_id']),
                    model_name=row['model_name'],
                    pit_probability=float(row['pit_probability']),
                    decision_threshold=float(row['decision_threshold']),
                    pit_prediction=bool(row['pit_prediction']),
                    actual_pit=bool(row.get('actual_pit')),
                    is_correct=bool(row.get('actual_pit')) == bool(row['pit_prediction'])
                )
                session.add(pred)
            session.commit()
            print(f"✓ Inserted {len(predictions_df)} predictions")
        finally:
            session.close()

    def query_predictions(self, model_name: str, limit: int = 1000) -> pd.DataFrame:
        """Query recent predictions"""
        session = self.get_session()
        try:
            query = session.query(ModelPredictionORM).filter(
                ModelPredictionORM.model_name == model_name
            ).order_by(ModelPredictionORM.created_at.desc()).limit(limit)
            results = pd.read_sql(query.statement, self.engine)
            return results
        finally:
            session.close()

    def query_metrics(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Query model metrics"""
        session = self.get_session()
        try:
            query = session.query(ModelMetricsORM)
            if model_name:
                query = query.filter(ModelMetricsORM.model_name == model_name)
            results = pd.read_sql(query.statement, self.engine)
            return results
        finally:
            session.close()

    def insert_metrics(self, metrics_dict: dict):
        """Insert model evaluation metrics"""
        session = self.get_session()
        try:
            metrics = ModelMetricsORM(**metrics_dict)
            session.add(metrics)
            session.commit()
            print(f"✓ Inserted metrics for {metrics_dict['model_name']}")
        finally:
            session.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # PostgreSQL
    connector = SQLConnector(
        'postgresql',
        user='postgres',
        password='password',
        host='localhost',
        database='f1_pit_db'
    )

    # Initialize database
    connector.init_db()

    # Example: Insert metrics
    metrics = {
        'model_name': 'Random Forest',
        'test_set_name': 'test_2024',
        'accuracy': 0.6337,
        'precision': 0.2826,
        'recall': 0.9163,
        'f1_score': 0.4320,
        'roc_auc': 0.7600,
        'pr_auc': 0.2687,
        'mae': 0.2174,
        'rmse': 0.3848,
        'r2': 0.3240,
        'decision_threshold': 0.60,
        'threshold_type': 'conservative',
        'num_trees': 100,
        'max_depth': 10
    }
    connector.insert_metrics(metrics)

    # Example: Query predictions
    predictions = connector.query_predictions('Random Forest', limit=100)
    print(f"Recent predictions:\n{predictions.head()}")
