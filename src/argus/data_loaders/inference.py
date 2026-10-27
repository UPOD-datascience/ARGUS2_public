import datetime
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from argus.utils.config import load_config
from argus.utils.logger import Logger

from .base import (
    DecursusDataLoader,
    ECGDataLoader,
    InferenceBaseDataLoader,
    LabDataLoader,
    MeasurementDataLoader,
    PatselDataLoader,
)

load_dotenv()


class InferencePatselLoader(InferenceBaseDataLoader, PatselDataLoader):
    """Loads patsel data from database for inference"""

    def __init__(
        self,
        logger: Logger,
        version: str,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        from_date: Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize InferencePatselLoader"""
        # Get database credentials - use provided or fall back to .env
        db_user = db_user or os.getenv("DB_USER")
        db_password = db_password or os.getenv("DB_PASSWD")

        if not db_user or not db_password:
            raise ValueError("Database credentials not found. Set DB_USER and DB_PASSWD ")

        # Load patsel_processor config for database and column settings
        processor_config = load_config("patsel_processor", version=version, mode="inference")

        # Get database config
        db_config = processor_config.get("database", {})
        host = db_config.get("host", "dataplatform-daily")
        database = db_config.get("database", "SELF_SERVICE")

        self.from_date = from_date

        super().__init__(
            logger=logger,
            version=version,
            mode="inference",
            db_user=db_user,
            db_password=db_password,
            host=host,
            database=database,
        )

        self.schema = db_config.get("schema", "argus")

        # Create the database engine
        self._create_engine()

        # Columns to select from database
        self.columns_to_select = processor_config.get("patsel_columns", [])

        if not self.columns_to_select:
            raise ValueError(
                f"patsel_columns not found in patsel_processor config for version {version}, mode inference"
            )

    def load_raw_data(self) -> pd.DataFrame:
        """Load from database"""
        self.logger.info(f"Loading patsel data from [{self.database}].[{self.schema}].[PatSel]")

        # Build SELECT statement
        columns_str = ", ".join(self.columns_to_select)

        # if self.from_date is not None add date where clause
        if self.from_date is not None:
            where_clause = f"WHERE Index_dt >= '{self.from_date.isoformat()}'"
        else:
            where_clause = ""

        query = f"""
            SELECT {columns_str}
            FROM [{self.database}].[{self.schema}].[PatSel]
            {where_clause}
            OPTION (MAXDOP 1);
        """

        self.logger.info(f"Executing query: {query}")

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(query), connection)

            self.logger.info(f"Loaded {len(df)} raw records from database")
            self.logger.info(f"Columns: {list(df.columns)}")

            # Convert Index_dt to datetime
            index_date_col = self.columns_config["index_date"]
            if index_date_col in df.columns:
                df[index_date_col] = pd.to_datetime(df[index_date_col])
                self.logger.info(f"Converted {index_date_col} to datetime")

            return df

        except Exception as e:
            self.logger.error(f"Failed to load data from database: {e}")
            raise


class InferenceDecursusLoader(InferenceBaseDataLoader, DecursusDataLoader):
    """Loads decursus data from database for inference"""

    def __init__(
        self,
        logger: Logger,
        version: str,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        debug_mode: bool = False,
        from_date: Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize InferenceDecursusLoader"""
        db_user = db_user or os.getenv("DB_USER")
        db_password = db_password or os.getenv("DB_PASSWD")

        if not db_user or not db_password:
            raise ValueError("Database credentials not found")

        # Load patsel_processor config for database settings
        processor_config = load_config("patsel_processor", version=version, mode="inference")

        # Get database config from processor config
        db_config = processor_config.get("database", {})
        host = db_config.get("host", "dataplatform-daily")
        database = db_config.get("database", "SELF_SERVICE")

        self.from_date = from_date

        super().__init__(
            logger=logger,
            version=version,
            mode="inference",
            db_user=db_user,
            db_password=db_password,
            host=host,
            database=database,
        )

        self.debug_mode = debug_mode
        self.schema = db_config.get("schema", "argus")

        # Create the database engine
        self._create_engine()

        # Columns to select
        self.columns_to_select = processor_config.get("decursus_columns", [])

    def load_raw_data(self) -> pd.DataFrame:
        """Load decursus from database"""
        self.logger.info(f"Loading decursus data from [{self.database}].[{self.schema}].[Consult]")

        if self.from_date is not None:
            where = f"WHERE [Index_dt] >= '{self.from_date.isoformat()}'"
        else:
            where = ""

        if self.debug_mode:
            self.logger.info("TEST - Run in debug mode")
            query = f"""
                SELECT TOP (50) *
                FROM [{self.database}].[{self.schema}].[Consult]
                {where}
                OPTION (MAXDOP 1);
            """
        elif self.columns_to_select:
            columns_str = ", ".join(self.columns_to_select)
            query = f"""
                SELECT {columns_str}
                FROM [{self.database}].[{self.schema}].[Consult]
                {where}
                OPTION (MAXDOP 1);
            """
        else:
            query = f"""
                SELECT *
                FROM [{self.database}].[{self.schema}].[Consult]
                {where}
                OPTION (MAXDOP 1);
            """

        self.logger.info(f"Executing query: {query}")

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(query), connection)

            self.logger.info(f"Loaded {len(df)} decursus records from database")

            # Convert studyId to string
            general_config = load_config("general", version=self.version, mode=self.mode)
            study_id_col = general_config["columns"]["study_id"]

            if study_id_col in df.columns:
                if df[study_id_col].dtype != "object":
                    df[study_id_col] = df[study_id_col].astype(int)  # Remove floats
                    df[study_id_col] = df[study_id_col].astype(str)
                    self.logger.info(f"Converted {study_id_col} to string type")

            # Drop nulls in section_text
            section_text_col = self.processor_config.get("columns", {}).get("section_text_col")
            if section_text_col and section_text_col in df.columns:
                initial_count = len(df)
                df = df.dropna(subset=[section_text_col])
                dropped = initial_count - len(df)
                if dropped > 0:
                    self.logger.info(f"Dropped {dropped} rows with null {section_text_col}")

            return df

        except Exception as e:
            self.logger.error(f"Failed to load decursus from database: {e}")
            raise


class InferenceLabLoader(InferenceBaseDataLoader, LabDataLoader):
    """Loads lab data from database for inference"""

    def __init__(
        self,
        logger: Logger,
        version: str,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        from_date: Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize InferenceLabLoader"""
        db_user = db_user or os.getenv("DB_USER")
        db_password = db_password or os.getenv("DB_PASSWD")

        if not db_user or not db_password:
            raise ValueError("Database credentials not found")

        # Load general config for database settings
        general_config = load_config("general", version=version, mode="inference")

        # Get database config
        db_config = general_config.get("database", {})
        host = db_config.get("host", "dataplatform-daily")
        database = db_config.get("database", "SELF_SERVICE")

        self.from_date = from_date

        super().__init__(
            logger=logger,
            version=version,
            mode="inference",
            db_user=db_user,
            db_password=db_password,
            host=host,
            database=database,
        )

        self.schema = db_config.get("schema", "argus")

        # Create the database engine
        self._create_engine()

        # Columns to select
        self.columns_to_select = general_config.get("lab_columns", [])

    def load_raw_data(self) -> pd.DataFrame:
        """Load lab data from database"""
        self.logger.info(f"Loading lab data from [{self.database}].[{self.schema}].[Lab]")

        if self.from_date is not None:
            where = f" WHERE [Index_dt] >= '{self.from_date.isoformat()}'"
        else:
            where = ""

        if self.columns_to_select:
            columns_str = ", ".join(self.columns_to_select)
            query = f"""
                SELECT {columns_str}
                FROM [{self.database}].[{self.schema}].[Lab]
                {where}
                OPTION (MAXDOP 1);
            """
        else:
            query = f"""
                SELECT *
                FROM [{self.database}].[{self.schema}].[Lab]
                {where}
                OPTION (MAXDOP 1);
            """

        self.logger.info(f"Executing query: {query}")

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(query), connection)

            self.logger.info(f"Loaded {len(df)} lab records from database")

            # Convert studyId to string (to match patsel data type)
            general_config = load_config("general", version=self.version, mode=self.mode)
            study_id_col = general_config["columns"]["study_id"]

            if study_id_col in df.columns:
                if df[study_id_col].dtype != "object":
                    df[study_id_col] = df[study_id_col].astype(int)  # Remove floats
                    df[study_id_col] = df[study_id_col].astype(str)
                    self.logger.info(f"Converted {study_id_col} to string type")

            return df

        except Exception as e:
            self.logger.error(f"Failed to load lab data from database: {e}")
            raise


class InferenceECGLoader(InferenceBaseDataLoader, ECGDataLoader):
    """Loads ECG data from database for inference"""

    def __init__(
        self,
        logger: Logger,
        version: str,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        ecg_type: str = "classresult",
        from_date: Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize InferenceECGLoader"""
        # Validate ecg_type
        if ecg_type not in ["classresult", "detail"]:
            raise ValueError(f"ecg_type must be 'classresult' or 'detail', got '{ecg_type}'")

        self.ecg_type = ecg_type

        db_user = db_user or os.getenv("DB_USER")
        db_password = db_password or os.getenv("DB_PASSWD")

        if not db_user or not db_password:
            raise ValueError("Database credentials not found")

        # Load general config for database settings
        general_config = load_config("general", version=version, mode="inference")

        # Get database config
        db_config = general_config.get("database", {})
        host = db_config.get("host", "dataplatform-daily")
        database = db_config.get("database", "SELF_SERVICE")

        self.from_date = from_date

        super().__init__(
            logger=logger,
            version=version,
            mode="inference",
            db_user=db_user,
            db_password=db_password,
            host=host,
            database=database,
        )

        self.schema = db_config.get("schema", "argus")

        # Create the database engine
        self._create_engine()

        if ecg_type == "classresult":
            self.table_name = "ECG_classresult"
        elif ecg_type == "detail":
            self.table_name = "ECG_Detail"

    def load_raw_data(self) -> pd.DataFrame:
        """Load ECG data from database"""
        self.logger.info(f"Loading ECG {self.ecg_type} data from [{self.database}].[{self.schema}].[{self.table_name}]")

        if self.from_date is not None:
            where = f"WHERE [Index_dt] >= '{self.from_date.isoformat()}'"
        else:
            where = ""

        # For ECG data, select all columns
        query = f"""
            SELECT *
            FROM [{self.database}].[{self.schema}].[{self.table_name}]
            {where}
            OPTION (MAXDOP 1);
        """

        self.logger.info(f"Executing query: {query}")

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(query), connection)

            self.logger.info(f"Loaded {len(df)} ECG {self.ecg_type} records from database")

            # Make sure studyId names match and convert studyId to string (to match patsel data type)
            general_config = load_config("general", version=self.version, mode=self.mode)
            ecg_config = load_config("ecg", version=self.version, mode=self.mode)

            study_id_col = general_config["columns"]["study_id"]
            study_id_col_orig = ecg_config["columns_ecg"]["study_id"]

            if study_id_col_orig in df.columns and study_id_col_orig != study_id_col:
                df = df.rename(columns={study_id_col_orig: study_id_col})

            if study_id_col in df.columns:
                if df[study_id_col].dtype != "object":
                    df[study_id_col] = df[study_id_col].astype(int)  # Remove floats
                    df[study_id_col] = df[study_id_col].astype(str)
                    self.logger.info(f"Converted {study_id_col} to string type")

            # Drop duplicate rows
            df = df.drop_duplicates()

            return df

        except Exception as e:
            self.logger.error(f"Failed to load ECG {self.ecg_type} data from database: {e}")
            raise


class InferenceMeasurementLoader(InferenceBaseDataLoader, MeasurementDataLoader):
    """Loads measurement data (Body and Blood Pressure) from database for inference"""

    def __init__(
        self,
        logger: Logger,
        version: str,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        data_type: str = "body",
        from_date: Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize InferenceMeasurementLoader"""
        # Validate data_type
        if data_type not in ["body", "BP"]:
            raise ValueError(f"data_type must be 'body' or 'BP', got '{data_type}'")

        self.data_type = data_type

        db_user = db_user or os.getenv("DB_USER")
        db_password = db_password or os.getenv("DB_PASSWD")

        if not db_user or not db_password:
            raise ValueError("Database credentials not found")

        # Load general config for database settings
        general_config = load_config("general", version=version, mode="inference")

        # Get database config
        db_config = general_config.get("database", {})
        host = db_config.get("host", "dataplatform-daily")
        database = db_config.get("database", "SELF_SERVICE")

        self.from_date = from_date

        super().__init__(
            logger=logger,
            version=version,
            mode="inference",
            db_user=db_user,
            db_password=db_password,
            host=host,
            database=database,
        )

        self.schema = db_config.get("schema", "argus")

        # Create the database engine
        self._create_engine()

        if data_type == "body":
            self.table_name = "Meting_Mirka"
        elif data_type == "BP":
            self.table_name = "Meting_BP_Mirka"

    def load_raw_data(self) -> pd.DataFrame:
        """Load measurement data from database"""
        self.logger.info(
            f"Loading Measurement {self.data_type} data from [{self.database}].[{self.schema}].[{self.table_name}]"
        )

        if self.from_date is not None:
            where = f"WHERE [Index_dt] >= '{self.from_date.isoformat()}'"
        else:
            where = ""

        # For Measurement data, select all columns
        query = f"""
            SELECT *
            FROM [{self.database}].[{self.schema}].[{self.table_name}]
            {where}
            OPTION (MAXDOP 1);
        """

        self.logger.info(f"Executing query: {query}")

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(query), connection)

            self.logger.info(f"Loaded {len(df)} Measurement {self.data_type} records from database")

            # Make sure studyId names match and convert studyId to string (to match patsel data type)
            general_config = load_config("general", version=self.version, mode=self.mode)
            meting_config = load_config("metingen", version=self.version, mode=self.mode)

            study_id_col = general_config["columns"]["study_id"]
            study_id_col_orig = meting_config["columns"]["study_id"]

            if study_id_col_orig in df.columns and study_id_col_orig != study_id_col:
                df = df.rename(columns={study_id_col_orig: study_id_col})

            if study_id_col in df.columns:
                if df[study_id_col].dtype != "object":
                    df[study_id_col] = df[study_id_col].astype(int)  # Remove floats
                    df[study_id_col] = df[study_id_col].astype(str)
                    self.logger.info(f"Converted {study_id_col} to string type")

            # Make sure index column name from all datasets match
            index_col = general_config["columns"]["index_date"]
            index_col_orig = meting_config["columns"]["index_date"]

            if index_col_orig in df.columns and index_col_orig != index_col:
                df = df.rename(columns={index_col_orig: index_col})

            # Drop duplicate rows
            df = df.drop_duplicates()

            return df

        except Exception as e:
            self.logger.error(f"Failed to load Measurement {self.data_type} data from database: {e}")
            raise
