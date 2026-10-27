"""Database utilities for loading analysis data from ARGUS SQL databases."""

import os
import platform
import urllib.parse
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def create_engine_with_retry(connection_string: str, max_retries: int = 3) -> Engine:
    """Create a database engine with retry logic.

    Args:
        connection_string: Database connection string.
        max_retries: Maximum number of connection attempts.

    Returns:
        SQLAlchemy Engine instance.

    """
    for attempt in range(max_retries):
        try:
            engine = create_engine(
                connection_string, pool_pre_ping=True, pool_recycle=3600, echo=False, isolation_level="AUTOCOMMIT"
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise

    raise ConnectionError(f"Failed to connect after {max_retries} attempts")


def read_from_db(
    query: str, connection_string: Optional[str] = None, params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Read data from a database using a SQL query.

    Args:
        query: SQL query string.
        connection_string: Optional connection string. Uses the default if None.
        params: Optional query parameters.

    Returns:
        DataFrame with query results.

    """
    try:
        conn_string = connection_string
        engine = create_engine_with_retry(conn_string)

        df = pd.read_sql_query(query, con=engine, params=params)

        engine.dispose()

        return df

    except Exception as e:
        print(f"Error reading from database: {str(e)}")
        raise


def create_connection_string(db_name):
    load_dotenv()

    # Get database credentials - use provided or fall back to .env
    db_user = os.getenv("DB_USER")
    db_passwd = os.getenv("DB_PASSWD")
    db_port = os.getenv("DB_POSIT_PORT")

    if db_name == "SELF_SERVICE":
        # connect to Views in DPF-daily
        db_host = os.getenv("DB_HOST")
    elif db_name == os.getenv("DB_POSIT_DATABASE_ACC"):
        # connect to Results or Logging in Posit
        db_host = os.getenv("DB_POSIT_HOST_ACC")
    else:
        print(Warning(f"db_name expected to be 'SELF_SERVICE' or 'DB_POSIT_DATABASE_ACC', but is {db_name}"))

    # URL encode the username and password to handle special characters
    db_user_encoded = urllib.parse.quote_plus(db_user)
    db_passwd_encoded = urllib.parse.quote_plus(db_passwd)

    if db_user == "":
        print("Using debug SQLite database...")
        print("sqlite:///./sql_app.db", {"schema_translate_map": {"noshow": None}})

    if platform.system() in "Windows":
        # Windows: Use Windows authentication
        connection_string = (
            "mssql+pyodbc://?"
            "odbc_connect="
            "Driver={SQL Server};"
            f"Server={db_host},{db_port};"
            f"Database={db_name};"
            "Trusted_Connection=yes;"
        )
    else:
        # Non-Windows: Use SQL Server authentication with pymssql
        # pymssql is more compatible with macOS/Linux
        connection_string = f"mssql+pymssql://{db_user_encoded}:{db_passwd_encoded}@{db_host}:{db_port}/{db_name}"

    return connection_string


def load_argus_inference_sql(query: str, db_name: str):  # noqa: D103
    df = read_from_db(query, create_connection_string(db_name))

    return df
