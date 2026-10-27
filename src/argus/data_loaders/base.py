from abc import ABC, abstractmethod
from typing import Type

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

from argus.utils.config import load_config
from argus.utils.logger import Logger


class BaseDataLoader(ABC):
    """Abstract base for all data loaders."""

    def __init__(self, logger: "Logger", version: str, mode: str):
        self.logger = logger
        self.version = version
        self.mode = mode

        # Load mode-specific configuration
        self.config = load_config("general", version=version, mode=mode)  # Add mode here!

        # Get columns (mode-specific)
        self.columns_config = self.config["columns"]
        self.expected_types = self.config.get("expected_types", {})

        # Training-specific setup
        if mode == "train":
            self.data_sources = self.config.get("data_sources", {})
            self.folder = self.data_sources.get("folder")
            self.subfolder = self.data_sources.get("subfolder")

    def get_file_config(self, file_key: str) -> str:
        """Get filename from config by key."""
        filename = self.data_sources.get(file_key)
        if filename is None:
            raise ValueError(f"File key '{file_key}' not found in config for version {self.version}")
        return filename

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data - implemented differently per mode and database"""
        pass

    @abstractmethod
    def get_schema_class(self) -> Type["BaseModel"]:
        """Return the Pydantic schema class for validation"""
        pass

    def validate_output(self, df: pd.DataFrame) -> "BaseModel":
        """Validate output using the appropriate Pydantic schema"""
        schema_class = self.get_schema_class()
        return schema_class.from_dataframe(df, version=self.version, mode=self.mode)

    def load_data(self) -> pd.DataFrame:
        """Template method: load and validate"""
        try:
            self.logger.info(f"Loading data using {self.__class__.__name__}")
            raw_data = self.load_raw_data()

            self.logger.info("Validating output schema...")
            validated = self.validate_output(raw_data)

            self.logger.info(
                f"Successfully loaded {validated.record_count} records with {len(validated.columns)} columns"
            )

            return validated.data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise


class InferenceBaseDataLoader:
    """Mixin class for inference loaders with database functionality"""

    def __init__(self, db_user: str = None, db_password: str = None, host: str = None, database: str = None, **kwargs):
        """Initialize database connection parameters"""
        super().__init__(**kwargs)

        self.db_user = db_user
        self.db_password = db_password
        self.host = host
        self.database = database

        # if self.db_user and self.db_password and self.host and self.database:
        #    self._create_engine()

    def _create_engine(self):
        """Create SQLAlchemy engine"""
        db_url = URL.create(
            drivername="mssql+pymssql",
            username=self.db_user,
            password=self.db_password,
            host=self.host,
            database=self.database,
        )
        self.engine = create_engine(db_url)
        self.logger.info(f"Created database engine for {self.host}/{self.database}")

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, "engine"):
            self.engine.dispose()


class PatselDataLoader(BaseDataLoader):
    """Base class specifically for patsel data with processing"""

    def __init__(self, logger: "Logger", version: str, mode: str):
        super().__init__(logger, version, mode)
        # Column names automatically selected based on mode
        self.studyId = self.columns_config["study_id"]
        self.index_date = self.columns_config["index_date"]

    def get_schema_class(self):
        from argus.data_models.patsel import PatselDataSchema

        return PatselDataSchema

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Common processing logic for patsel data"""
        from argus.ETL.ETL_patsel import PatientSelector

        processor = PatientSelector(raw_data, self.logger, self.version, self.mode)
        return processor.process_patient_selection_data()

    def load_data(self) -> pd.DataFrame:
        """Override to include processing step"""
        try:
            self.logger.info(f"Loading patsel data using {self.__class__.__name__}")
            raw_data = self.load_raw_data()

            self.logger.info("Processing patsel data...")
            processed_data = self.process_data(raw_data)

            self.logger.info("Validating output schema...")
            validated = self.validate_output(processed_data)

            self.logger.info(
                f"Successfully loaded {validated.record_count} records with {len(validated.columns)} columns"
            )

            return validated.data

        except Exception as e:
            self.logger.error(f"Error loading/processing patsel data: {e}")
            raise


class DecursusDataLoader(BaseDataLoader):
    """Base class for decursus data"""

    def __init__(self, logger: "Logger", version: str, mode: str):
        super().__init__(logger, version, mode)

        # Load patsel_processor config for column names
        self.processor_config = load_config("patsel_processor", version=version, mode=mode)

    def get_schema_class(self):
        from argus.data_models.decursus import DecursusDataSchema

        return DecursusDataSchema


class LabDataLoader(BaseDataLoader):
    """Base class for Lab data"""

    def __init__(self, logger: "Logger", version: str, mode: str):
        super().__init__(logger, version, mode)

        # Load lab config for column names
        self.processor_config = load_config("lab", version=version, mode=mode)

    def get_schema_class(self):
        from argus.data_models.lab import LabDataSchema

        return LabDataSchema


class ECGDataLoader(BaseDataLoader):
    """Base class for ECG data"""

    def __init__(self, logger: "Logger", version: str, mode: str):
        super().__init__(logger, version, mode)

        # Load ecg config for column names
        self.processor_config = load_config("ecg", version=version, mode=mode)

    def get_schema_class(self):
        from argus.data_models.ecg import ECGDataSchema

        return ECGDataSchema


class MeasurementDataLoader(BaseDataLoader):
    """Base class for Metingen (Body and Bloodpressure) data"""

    def __init__(self, logger: "Logger", version: str, mode: str):
        super().__init__(logger, version, mode)

        # Load ecg config for column names
        self.processor_config = load_config("metingen", version=version, mode=mode)

    def get_schema_class(self):
        from argus.data_models.meting import MeasurementsDataSchema

        return MeasurementsDataSchema
