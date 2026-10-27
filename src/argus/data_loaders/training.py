import pandas as pd
import pyreadstat

from argus.utils.config import load_config
from argus.utils.logger import Logger
from argus.utils.PathManager import PathManager_RES_Folder_UPOD

from .base import (
    DecursusDataLoader,
    ECGDataLoader,
    LabDataLoader,
    MeasurementDataLoader,
    PatselDataLoader,
)


class TrainingPatselLoader(PatselDataLoader):
    """Loads patsel data from local files for training."""

    def __init__(self, logger: Logger, version: str, path_manager: PathManager_RES_Folder_UPOD):
        super().__init__(logger, version, mode="train")
        self.path_manager = path_manager

    def load_raw_data(self) -> pd.DataFrame:
        """Load from SAS file using config."""
        self.logger.info("Loading raw patsel data from SAS file...")

        # Get file info from config
        patsel_filename = self.get_file_config("patsel_file")

        # Build path using config values
        path_patsel = self.path_manager.get_file_path(
            self.folder,  # From config
            self.subfolder,  # From config
            patsel_filename,  # From config
        )

        self.logger.info(f"Reading from: {path_patsel}")
        self.logger.info(f"  Folder: {self.folder}")
        self.logger.info(f"  Subfolder: {self.subfolder}")
        self.logger.info(f"  File: {patsel_filename}")

        dfPatsel = pyreadstat.read_sas7bdat(path_patsel)[0]

        self.logger.info(f"Loaded {len(dfPatsel)} raw records")
        return dfPatsel


class TrainingDecursusLoader(DecursusDataLoader):
    """Loads decursus data from local files for training."""

    def __init__(self, logger: Logger, version: str, path_manager: PathManager_RES_Folder_UPOD):
        super().__init__(logger, version, mode="train")
        self.path_manager = path_manager

    def load_raw_data(self) -> pd.DataFrame:
        """Load decursus from local file using config."""
        self.logger.info("Loading raw decursus data from file...")

        # Get file paths from general config (same as patsel)
        decursus_filename = self.get_file_config("decursus_file")

        # Get encoding from processor config
        encoding = self.processor_config.get("decursus_encoding", "iso-8859-1")

        # Build path using config values from general config
        path_decursus = self.path_manager.get_file_path(
            self.folder,  # From BaseDataLoader (general config)
            self.subfolder,  # From BaseDataLoader (general config)
            decursus_filename,  # From general config
        )

        self.logger.info(f"Reading from: {path_decursus} with encoding: {encoding}")

        # Read file
        df = pyreadstat.read_sas7bdat(path_decursus, encoding=encoding)[0]

        self.logger.info(f"Loaded {len(df)} raw decursus records")

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


class TrainingLabLoader(LabDataLoader):
    """Loads lab data from local files for training"""

    def __init__(self, logger: Logger, version: str, path_manager: PathManager_RES_Folder_UPOD):
        super().__init__(logger, version, mode="train")
        self.path_manager = path_manager

    def load_raw_data(self) -> pd.DataFrame:
        """Load lab from local file using config"""
        self.logger.info("Loading raw lab data from file...")

        # Get file paths from general config
        lab_filename = self.get_file_config("lab_file")

        # Build path using config values from general config
        path_lab = self.path_manager.get_file_path(
            self.folder,  # From BaseDataLoader (general config)
            self.subfolder,  # From BaseDataLoader (general config)
            lab_filename,  # From general config
        )

        self.logger.info(f"Reading from: {path_lab}")

        # Read file
        df = pyreadstat.read_sas7bdat(path_lab)[0]

        self.logger.info(f"Loaded {len(df)} raw lab records")

        general_config = load_config("general", version=self.version, mode=self.mode)
        study_id_col = general_config["columns"]["study_id"]

        if study_id_col in df.columns:
            if df[study_id_col].dtype != "object":
                df[study_id_col] = df[study_id_col].astype(int)  # Remove floats
                df[study_id_col] = df[study_id_col].astype(str)
                self.logger.info(f"Converted {study_id_col} to string type")

        return df


class TrainingECGLoader(ECGDataLoader):
    """Loads ECG data from local files for training"""

    def __init__(
        self,
        logger: Logger,
        version: str,
        path_manager: PathManager_RES_Folder_UPOD,
        ecg_type: str = "classresult",  # "classresult" or "detail"
    ):
        super().__init__(logger, version, mode="train")
        self.path_manager = path_manager
        self.ecg_type = ecg_type

        # Validate ecg_type
        if ecg_type not in ["classresult", "detail"]:
            raise ValueError(f"ecg_type must be 'classresult' or 'detail', got '{ecg_type}'")

    def load_raw_data(self) -> pd.DataFrame:
        """Load ECG data from local file based on ecg_type"""
        self.logger.info(f"Loading ECG {self.ecg_type} data...")

        # Get filename based on type
        if self.ecg_type == "classresult":
            ecg_filename = self.get_file_config("ecg_file1")
        elif self.ecg_type == "detail":
            ecg_filename = self.get_file_config("ecg_file2")

        # Build path
        path_ecg = self.path_manager.get_file_path(self.folder, self.subfolder, ecg_filename)

        self.logger.info(f"Reading from: {path_ecg}")
        df = pyreadstat.read_sas7bdat(path_ecg)[0]
        self.logger.info(f"Loaded {len(df)} ECG {self.ecg_type} records")

        # Convert studyId to string
        general_config = load_config("general", version=self.version, mode=self.mode)
        ecg_config = load_config("ecg", version=self.version, mode=self.mode)

        study_id_col = general_config["columns"]["study_id"]
        if study_id_col in df.columns and df[study_id_col].dtype != "object":
            df[study_id_col] = df[study_id_col].astype(int).astype(str)
            self.logger.info(f"Converted {study_id_col} to string type")

            # drop duplicate rows
            df = df.drop_duplicates()

        return df


class TrainingMeasurementLoader(MeasurementDataLoader):
    """Loads Measurement data from local files for training"""

    def __init__(
        self,
        logger: Logger,
        version: str,
        path_manager: PathManager_RES_Folder_UPOD,
        data_type: str = "body",  # "body" or "BP"
    ):
        super().__init__(logger, version, mode="train")
        self.path_manager = path_manager
        self.data_type = data_type

        # Validate ecg_type
        if data_type not in ["body", "BP"]:
            raise ValueError(f"data_type must be 'body' or 'BP', got '{data_type}'")

    def load_raw_data(self) -> pd.DataFrame:
        """Load Measurement data from local file based on data_type"""
        self.logger.info(f"Loading Measurement {self.data_type} data...")

        # Get filename based on type
        if self.data_type == "body":
            filename = self.get_file_config("body_file")
        elif self.data_type == "BP":
            filename = self.get_file_config("BP_file")

        # Build path
        path = self.path_manager.get_file_path(self.folder, self.subfolder, filename)

        self.logger.info(f"Reading from: {path}")
        df = pyreadstat.read_sas7bdat(path)[0]
        self.logger.info(f"Loaded {len(df)} {self.data_type} measurement records")

        # Convert studyId to string
        general_config = load_config("general", version=self.version, mode=self.mode)
        measurements_config = load_config("metingen", version=self.version, mode=self.mode)

        study_id_col = general_config["columns"]["study_id"]
        if study_id_col in df.columns and df[study_id_col].dtype != "object":
            df[study_id_col] = df[study_id_col].astype(int).astype(str)
            self.logger.info(f"Converted {study_id_col} to string type")

            # drop duplicate rows
            df = df.drop_duplicates()

        return df
