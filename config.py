"""
Configuration Management for F1 Pit Strategy System

Uses Pydantic BaseSettings to load configuration from environment variables.
Supports multiple environments (development, staging, production).

Usage:
    from config import settings
    print(settings.db_host)
    print(settings.environment)

Setup:
    1. Copy .env.example to .env
    2. Update values in .env as needed
    3. config.py automatically loads from .env in the working directory
"""

from pydantic import BaseSettings, Field
from typing import Literal
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    Attributes:
        db_type: Database type (postgresql, mysql, sqlserver)
        db_user: Database username
        db_password: Database password
        db_host: Database hostname
        db_port: Database port
        db_name: Database name
        environment: Environment (development, staging, production)
        streamlit_port: Streamlit server port
        streamlit_address: Streamlit server address
        streamlit_logger_level: Streamlit logger level
        fastf1_api_timeout: FastF1 API timeout in seconds
        fastf1_cache_dir: FastF1 cache directory
        log_level: Logging level
        log_format: Logging format (text or json)
    """

    # Database Configuration
    db_type: Literal["postgresql", "mysql", "sqlserver"] = Field(
        default="postgresql",
        description="Database type"
    )
    db_user: str = Field(
        default="postgres",
        description="Database username"
    )
    db_password: str = Field(
        default="",
        description="Database password (KEEP SECURE!)"
    )
    db_host: str = Field(
        default="localhost",
        description="Database hostname"
    )
    db_port: int = Field(
        default=5432,
        description="Database port"
    )
    db_name: str = Field(
        default="f1_pit_db",
        description="Database name"
    )

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Environment type"
    )

    # Streamlit Configuration
    streamlit_port: int = Field(
        default=8501,
        description="Streamlit server port"
    )
    streamlit_address: str = Field(
        default="0.0.0.0",
        description="Streamlit server address"
    )
    streamlit_logger_level: str = Field(
        default="info",
        description="Streamlit logger level"
    )

    # FastF1 API Configuration
    fastf1_api_timeout: int = Field(
        default=10,
        description="FastF1 API timeout in seconds"
    )
    fastf1_cache_dir: str = Field(
        default=".cache/",
        description="FastF1 cache directory"
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_format: Literal["text", "json"] = Field(
        default="json",
        description="Log format (text for development, json for production)"
    )

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = "utf-8"

    def get_db_url(self) -> str:
        """
        Generate database connection URL for SQLAlchemy.

        Returns:
            str: Database connection string

        Raises:
            ValueError: If db_type is not supported
        """
        if self.db_type == "postgresql":
            return (
                f"postgresql://{self.db_user}:{self.db_password}@"
                f"{self.db_host}:{self.db_port}/{self.db_name}"
            )
        elif self.db_type == "mysql":
            return (
                f"mysql+pymysql://{self.db_user}:{self.db_password}@"
                f"{self.db_host}:{self.db_port}/{self.db_name}"
            )
        elif self.db_type == "sqlserver":
            return (
                f"mssql+pyodbc://{self.db_user}:{self.db_password}@"
                f"{self.db_host}/{self.db_name}?driver=ODBC+Driver+17+for+SQL+Server"
            )
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def __repr__(self) -> str:
        """String representation (hide password)"""
        db_url = self.get_db_url()
        # Mask password in URL
        masked_url = db_url.replace(self.db_password, "***")
        return (
            f"Settings(environment={self.environment}, "
            f"db_url={masked_url}, "
            f"log_level={self.log_level})"
        )


# Global settings instance (lazy-loaded)
try:
    settings = Settings()
    logger.info(f"✓ Configuration loaded: {settings}")
except Exception as e:
    logger.error(f"✗ Failed to load configuration: {e}")
    logger.error("Hint: Copy .env.example to .env and fill in your database credentials")
    raise


if __name__ == "__main__":
    """
    Print current configuration (useful for debugging).
    Run: python config.py
    """
    print("\n" + "=" * 80)
    print("F1 PIT STRATEGY - CONFIGURATION")
    print("=" * 80)
    print(f"\nEnvironment: {settings.environment}")
    print(f"Log Level: {settings.log_level}")
    print(f"Log Format: {settings.log_format}")
    print(f"\nDatabase Configuration:")
    print(f"  Type: {settings.db_type}")
    print(f"  Host: {settings.db_host}:{settings.db_port}")
    print(f"  Database: {settings.db_name}")
    print(f"  User: {settings.db_user}")
    print(f"  Password: {'*' * len(settings.db_password)}")
    print(f"\nStreamlit Configuration:")
    print(f"  Address: {settings.streamlit_address}:{settings.streamlit_port}")
    print(f"  Logger Level: {settings.streamlit_logger_level}")
    print(f"\nFastF1 Configuration:")
    print(f"  API Timeout: {settings.fastf1_api_timeout}s")
    print(f"  Cache Dir: {settings.fastf1_cache_dir}")
    print("\n" + "=" * 80)
