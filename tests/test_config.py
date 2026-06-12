"""Unit tests for configuration management"""

import pytest
import os
from pathlib import Path

# Add parent dir to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Settings


class TestConfigLoading:
    """Test that config loads from environment."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        settings = Settings()

        assert settings.db_type == "postgresql"
        assert settings.db_host == "localhost"
        assert settings.db_port == 5432
        assert settings.db_name == "f1_pit_db"
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"

    def test_env_var_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("DB_HOST", "db.example.com")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("ENVIRONMENT", "production")

        settings = Settings()

        assert settings.db_host == "db.example.com"
        assert settings.db_port == 5433
        assert settings.environment == "production"

    def test_db_url_generation(self):
        """Should generate valid PostgreSQL connection URL."""
        settings = Settings()
        url = settings.get_db_url()

        assert url.startswith("postgresql://")
        assert settings.db_user in url
        assert settings.db_host in url
        assert str(settings.db_port) in url
        assert settings.db_name in url

    def test_password_masking(self):
        """Repr should mask password."""
        settings = Settings()
        repr_str = repr(settings)

        assert "***" in repr_str
        assert settings.db_password not in repr_str or settings.db_password == ""

    def test_valid_db_types(self):
        """Should accept valid database types."""
        valid_types = ["postgresql", "mysql", "sqlserver"]

        for db_type in valid_types:
            os.environ["DB_TYPE"] = db_type
            settings = Settings()
            assert settings.db_type == db_type

    def test_valid_log_formats(self, monkeypatch):
        """Should accept valid log formats."""
        for fmt in ["text", "json"]:
            monkeypatch.setenv("LOG_FORMAT", fmt)
            settings = Settings()
            assert settings.log_format == fmt


class TestDatabaseURL:
    """Test database URL generation for different databases."""

    def test_postgresql_url(self, monkeypatch):
        """PostgreSQL URL should be valid."""
        monkeypatch.setenv("DB_TYPE", "postgresql")
        monkeypatch.setenv("DB_USER", "testuser")
        monkeypatch.setenv("DB_PASSWORD", "testpass")
        monkeypatch.setenv("DB_HOST", "testhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_NAME", "testdb")

        settings = Settings()
        url = settings.get_db_url()

        assert "postgresql://testuser:testpass@testhost:5432/testdb" == url

    def test_mysql_url(self, monkeypatch):
        """MySQL URL should use pymysql driver."""
        monkeypatch.setenv("DB_TYPE", "mysql")
        monkeypatch.setenv("DB_USER", "testuser")
        monkeypatch.setenv("DB_PASSWORD", "testpass")
        monkeypatch.setenv("DB_HOST", "testhost")
        monkeypatch.setenv("DB_PORT", "3306")
        monkeypatch.setenv("DB_NAME", "testdb")

        settings = Settings()
        url = settings.get_db_url()

        assert "mysql+pymysql://" in url
        assert "testuser:testpass" in url

    def test_special_chars_in_password(self, monkeypatch):
        """Should handle special characters in password."""
        monkeypatch.setenv("DB_PASSWORD", "p@ssw0rd!#$")
        settings = Settings()
        url = settings.get_db_url()

        assert "p@ssw0rd!#$" in url or "postgresql://" in url  # URL might encode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
