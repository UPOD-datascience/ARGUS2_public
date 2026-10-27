import logging
import os
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from argus.data_models.model_output import (
    get_column_mappings,
    get_expected_dtypes,
)


class DatabaseUtils:
    """Utility class for database operations and dataframe validation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize DatabaseUtils.

        Args:
            logger: Optional logger instance

        """
        self.logger = logger or logging.getLogger(__name__)
        self.expected_dtypes = get_expected_dtypes()

    def last_index(self, table_name: str = "shadow_test", date_column: str = "Index_dt", schema: str = "argus"):
        """Get the last index value from the specified table.

        Args:
            table_name: Name of the table to query
            date_column: Name of the date column to use for ordering

        Returns:
            Last index value as a datetime object

        """
        try:
            engine = self.create_engine_with_retry(self.get_posit_connection_string())
            with engine.connect() as conn:
                query = text(f"SELECT MAX({date_column}) FROM {schema}.{table_name}")
                result = conn.execute(query).fetchone()
                if result and result[0] is not None:
                    # Ensure the result is a datetime.datetime object
                    value = result[0]
                    if isinstance(value, datetime):
                        return value
                    else:
                        # Convert to datetime if it's a string or other type
                        return pd.to_datetime(value).to_pydatetime()
                return None
        except SQLAlchemyError as e:
            self.logger.error(f"Error fetching last index: {e}")
            return None

    def get_posit_connection_string(self, environment: Optional[str] = None) -> str:
        """Get the Posit Connect database connection string based on environment.

        Args:
            environment: Environment ('acc' or 'prod'). If None, uses ENV variable.

        Returns:
            Connection string for SQL Server

        """
        env = environment or os.getenv("ENVIRONMENT", "acc")

        if env == "prod":
            db_host = os.getenv("DB_POSIT_HOST_PROD")
            db_name = os.getenv("DB_POSIT_DATABASE_PROD")
        else:
            db_host = os.getenv("DB_POSIT_HOST_ACC")
            db_name = os.getenv("DB_POSIT_DATABASE_ACC")

        db_user = os.getenv("DB_USER")
        db_passwd = os.getenv("DB_PASSWD")
        db_port = os.getenv("DB_POSIT_PORT", "1433")

        if not all([db_host, db_name, db_user, db_passwd]):
            raise ValueError("Missing required database connection parameters in environment variables")

        # URL encode the username and password to handle special characters
        db_user_encoded = urllib.parse.quote_plus(db_user)
        db_passwd_encoded = urllib.parse.quote_plus(db_passwd)

        # Create connection string for SQL Server
        # Check if we're on Windows or not
        import platform

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

    def create_engine_with_retry(self, connection_string: str, max_retries: int = 3) -> Engine:
        """Create database engine with retry logic.

        Args:
            connection_string: Database connection string
            max_retries: Maximum number of connection attempts

        Returns:
            SQLAlchemy Engine instance

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
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise

        raise ConnectionError(f"Failed to connect after {max_retries} attempts")

    def validate_dataframe_schema(self, df: pd.DataFrame, strict: bool = False) -> Tuple[bool, List[str]]:
        """Validate dataframe against expected ModelOutput schema.

        Args:
            df: DataFrame to validate
            strict: If True, all expected columns must be present

        Returns:
            Tuple of (is_valid, list_of_issues)

        """
        issues = []

        if df.empty:
            return True, ["DataFrame is empty"]

        # Check for required columns
        required_columns = ["patientPseuID", "Age", "Index_dt"]
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")

        # Check for unexpected columns
        unexpected_columns = [col for col in df.columns if col not in self.expected_dtypes]
        if unexpected_columns and strict:
            issues.append(f"Unexpected columns: {unexpected_columns}")

        # Check data types for existing columns
        for col in df.columns:
            if col in self.expected_dtypes:
                expected_dtype = self.expected_dtypes[col]
                actual_dtype = str(df[col].dtype)

                # Allow some flexibility in type matching
                if not self._dtypes_compatible(actual_dtype, expected_dtype):
                    issues.append(f"Column '{col}' has type {actual_dtype}, expected {expected_dtype}")

        return len(issues) == 0, issues

    def _dtypes_compatible(self, actual: str, expected: str) -> bool:
        """Check if two dtype strings are compatible.

        Args:
            actual: Actual dtype string
            expected: Expected dtype string

        Returns:
            True if types are compatible

        """
        # Handle common type variations
        compatible_types = {
            "object": ["object", "string", "category"],
            "int64": ["int64", "int32", "int16", "int8", "Int64"],
            "float64": ["float64", "float32", "Float64"],
            "datetime64[ns]": ["datetime64[ns]", "datetime64[ns, UTC]"],
        }

        for base_type, variations in compatible_types.items():
            if expected == base_type:
                return any(actual.startswith(v) for v in variations)

        return actual == expected

    def prepare_dataframe_for_db(self, df: pd.DataFrame, add_metadata: bool = True) -> pd.DataFrame:
        """Prepare dataframe for database insertion.

        Args:
            df: DataFrame to prepare
            add_metadata: If True, add metadata columns (created_at, run_id)

        Returns:
            Prepared DataFrame

        """
        df_prepared = df.copy()

        # Convert data types where necessary
        for col in df_prepared.columns:
            if col in self.expected_dtypes:
                expected_dtype = self.expected_dtypes[col]

                try:
                    if expected_dtype == "datetime64[ns]":
                        df_prepared[col] = pd.to_datetime(df_prepared[col], errors="coerce")
                    elif expected_dtype == "float64":
                        df_prepared[col] = pd.to_numeric(df_prepared[col], errors="coerce")
                    elif expected_dtype == "int64":
                        # Handle integers with NaN values
                        df_prepared[col] = pd.to_numeric(df_prepared[col], errors="coerce")
                        if df_prepared[col].isna().any():
                            # Keep as float if there are NaN values
                            pass
                        else:
                            df_prepared[col] = df_prepared[col].astype("int64")
                    elif expected_dtype == "object":
                        df_prepared[col] = df_prepared[col].astype(str).replace("nan", None)
                except Exception as e:
                    self.logger.warning(f"Could not convert column '{col}' to {expected_dtype}: {str(e)}")

        # Only apply mappings for columns with spaces (backward compatibility)
        space_columns = {
            "HDL_Cholesterol": "HDL Cholesterol",
            "LDL_Cholesterol": "LDL Cholesterol",
            "eGFR_CKD_EPI": "eGFR (CKD-EPI)",
        }

        for old_name, new_name in space_columns.items():
            if old_name in df_prepared.columns and new_name not in df_prepared.columns:
                df_prepared.rename(columns={old_name: new_name}, inplace=True)

        # Replace infinity values with NaN
        df_prepared = df_prepared.replace([np.inf, -np.inf], np.nan)

        # Add metadata columns if requested
        if add_metadata:
            if "created_at" not in df_prepared.columns:
                df_prepared["created_at"] = datetime.now()
            if "run_id" not in df_prepared.columns:
                import uuid

                df_prepared["run_id"] = str(uuid.uuid4())

        return df_prepared

    def check_table_exists(self, engine: Engine, table_name: str, schema: str) -> bool:
        """Check if a table exists in the database.

        Args:
            engine: SQLAlchemy engine
            table_name: Name of the table
            schema: Database schema

        Returns:
            True if table exists

        """
        inspector = inspect(engine)
        return table_name in inspector.get_table_names(schema=schema)

    def create_table_if_not_exists(self, engine: Engine, table_name: str, schema: str, df_sample: pd.DataFrame):
        """Create table if it doesn't exist, using a sample dataframe for schema.

        Args:
            engine: SQLAlchemy engine
            table_name: Name of the table to create
            schema: Database schema
            df_sample: Sample dataframe to infer schema from

        """
        if not self.check_table_exists(engine, table_name, schema):
            try:
                # Use a connection with explicit transaction handling
                with engine.begin() as conn:
                    # Create empty table with schema from sample
                    df_empty = df_sample.iloc[:0].copy()
                    df_empty.to_sql(
                        name=table_name,
                        con=conn,  # Use connection, not engine
                        schema=schema,
                        if_exists="fail",
                        index=False,
                    )
                    self.logger.info(f"Created table {schema}.{table_name}")
            except Exception as e:
                self.logger.error(f"Failed to create table {schema}.{table_name}: {str(e)}")
                raise

    def write_dataframe_to_db(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str,
        connection_string: Optional[str] = None,
        if_exists: str = "append",
        validate: bool = True,
        prepare: bool = True,
        add_metadata: bool = True,
        chunksize: int = 1000,
    ) -> bool:
        """Write dataframe to database with validation and preparation.

        Args:
            df: DataFrame to write
            table_name: Target table name
            schema: Database schema
            connection_string: Optional connection string (uses default if None)
            if_exists: How to handle existing table ('append', 'replace', 'fail')
            validate: Whether to validate dataframe schema
            prepare: Whether to prepare dataframe for DB
            add_metadata: Whether to add metadata columns
            chunksize: Number of rows to write at once

        Returns:
            True if successful

        """
        try:
            # Get connection string
            conn_string = connection_string or self.get_posit_connection_string()

            # Create engine
            engine = self.create_engine_with_retry(conn_string)

            # Validate dataframe if requested
            if validate:
                is_valid, issues = self.validate_dataframe_schema(df)
                if not is_valid:
                    self.logger.warning(f"Validation issues found: {issues}")

            # Prepare dataframe if requested
            if prepare:
                df_to_write = self.prepare_dataframe_for_db(df, add_metadata=add_metadata)
            else:
                df_to_write = df.copy()
                if add_metadata:
                    df_to_write["created_at"] = datetime.now()
                    import uuid

                    df_to_write["run_id"] = str(uuid.uuid4())

            # Create table if it doesn't exist
            if if_exists != "replace":
                self.create_table_if_not_exists(engine, table_name, schema, df_to_write)

            # Write to database
            df_to_write.to_sql(
                name=table_name,
                con=engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=chunksize,
            )

            self.logger.info(f"Successfully wrote {len(df_to_write)} rows to {schema}.{table_name}")

            # Dispose of engine
            engine.dispose()

            return True

        except Exception as e:
            self.logger.error(f"Error writing to database: {str(e)}")
            raise

    def read_from_db(
        self, query: str, connection_string: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Read data from database using a SQL query.

        Args:
            query: SQL query string
            connection_string: Optional connection string (uses default if None)
            params: Optional query parameters

        Returns:
            DataFrame with query results

        """
        try:
            conn_string = connection_string or self.get_posit_connection_string()
            engine = self.create_engine_with_retry(conn_string)

            df = pd.read_sql_query(query, con=engine, params=params)

            engine.dispose()

            return df

        except Exception as e:
            self.logger.error(f"Error reading from database: {str(e)}")
            raise

    def migrate_column_names(self, df: pd.DataFrame, reverse: bool = False) -> pd.DataFrame:
        """Migrate dataframe column names between old and new naming conventions.

        Args:
            df: DataFrame with columns to migrate
            reverse: If True, migrate from new names to old names

        Returns:
            DataFrame with migrated column names

        """
        df_migrated = df.copy()
        column_mappings = get_column_mappings()

        if reverse:
            # Reverse the mappings (new to old)
            mappings_to_apply = {v: k for k, v in column_mappings.items() if k != v}
        else:
            # Normal mappings (old to new)
            mappings_to_apply = {k: v for k, v in column_mappings.items() if k != v}

        # Apply column renames
        for old_name, new_name in mappings_to_apply.items():
            if old_name in df_migrated.columns:
                df_migrated.rename(columns={old_name: new_name}, inplace=True)
                self.logger.debug(f"Renamed column '{old_name}' to '{new_name}'")

        return df_migrated
