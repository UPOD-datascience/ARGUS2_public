import datetime
import functools
import logging
import time
from typing import Optional

import pandas as pd

from argus.data_loaders.factory import (
    create_ecg_loader,
    create_lab_loader,
)
from argus.ETL.ETL_ecg import ECGdataProcessor
from argus.ETL.ETL_lab import LabDataProcessor
from argus.ETL.ETL_metingen import MetingenDataProcessor
from argus.utils.logger import Logger
from argus.utils.PathManager import PathManager_RES_Folder_UPOD


def timer_log_method(logger_attr: str = "logger"):
    """Decorator factory that logs duration to an instance logger if present,
    otherwise falls back to an appropriate module/class logger.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            # Try to find a logger on the first positional argument (usually `self`)
            obj = args[0] if args else None
            logger = None
            if obj is not None and hasattr(obj, logger_attr):
                logger = getattr(obj, logger_attr)
            if logger is None:
                # Fallback: prefer the object's module, else the function module
                name = obj.__class__.__module__ if obj is not None else func.__module__
                logger = logging.getLogger(name)
            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                # Use logger.info, but guard if logger missing method
                if hasattr(logger, "info"):
                    logger.info(f"{func.__name__} took {duration:.4f} seconds")
                else:
                    # Last-resort fallback to root logger
                    logging.getLogger().info(f"{func.__name__} took {duration:.4f} seconds")

        return wrapper

    return decorator


class LabProcessor:
    def __init__(
        self,
        logger: Logger,
        mode: str,  # 'train', 'inference'
        version: str = "v2_ARGUS25",
        path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
        from_date: Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize LabProcessor with configuration loading and validation"""
        # Set up logger
        self.logger = logger
        self.mode = mode  # 'train', 'inference'
        self.version = version

        # Create data loader using factory
        self.lab_loader = create_lab_loader(
            mode=self.mode, logger=self.logger, version=self.version, path_manager=path_manager, from_date=from_date
        )

    @timer_log_method(logger_attr="logger")
    def load_lab_data(self) -> pd.DataFrame:
        """Load lab data using the appropriate loader"""
        return self.lab_loader.load_data()

    @timer_log_method(logger_attr="logger")
    def process_lab(self, df_lab) -> pd.DataFrame:
        lab_processor = LabDataProcessor(df_lab, logger=self.logger, version=self.version, mode=self.mode)
        return lab_processor.process_all_features()

    @timer_log_method(logger_attr="logger")
    def main(self):
        dfLab = self.load_lab_data()
        df_processed_lab = self.process_lab(dfLab)
        return df_processed_lab


class ECGProcessor:
    def __init__(
        self,
        logger: Logger,
        mode: str,  # 'train', 'inference'
        version: str = "v2_ARGUS25",
        path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
        from_date: Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize ECGProcessor with configuration loading and validation"""
        # Set up logger
        self.logger = logger
        self.mode = mode  # 'train', 'inference'
        self.version = version
        self.path_manager = path_manager

        self.from_date = from_date

    @timer_log_method(logger_attr="logger")
    def load_ecg_data(self, ecg_type: str) -> pd.DataFrame:
        """Load ECG data of specified type"""
        loader = create_ecg_loader(
            mode=self.mode,
            logger=self.logger,
            version=self.version,
            ecg_type=ecg_type,
            path_manager=self.path_manager,
            from_date=self.from_date,
        )
        return loader.load_data()

    @timer_log_method(logger_attr="logger")
    def process_ecg(self, df_ecg_detail, df_ecg_classresult) -> pd.DataFrame:
        ecg_processor = ECGdataProcessor(df_ecg_detail.copy(), logger=self.logger, version=self.version, mode=self.mode)
        # Process ECG detail
        df_processed_ecg_detail = ecg_processor.process_ecg_data()

        # Merge with Class results ECG
        df_ecg_merged = ecg_processor.merge_ecg_detail_classresults(df_processed_ecg_detail, df_ecg_classresult)

        # Create composite ECG features:
        df_ecg_composites = ecg_processor.create_composites(df_ecg_merged)

        return df_ecg_composites

    @timer_log_method(logger_attr="logger")
    def main(self):
        df_ecg_classresult = self.load_ecg_data(ecg_type="classresult")
        df_ecg_detail = self.load_ecg_data(ecg_type="detail")
        df_ecg_merged = self.process_ecg(df_ecg_detail, df_ecg_classresult)

        return df_ecg_merged


class Body_and_BP_Processor:
    def __init__(
        self,
        logger: Logger,
        mode: str,  # 'train', 'inference'
        version: str = "v2_ARGUS25",
        path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
        from_date: Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize Body_and_BP_Processor (for body (height and weigth) and BloopPressure Measurements)
        with configuration loading and validation
        """
        # Set up logger
        self.logger = logger
        self.mode = mode  # 'train', 'inference'
        self.version = version
        self.path_manager = path_manager

        self.from_date = from_date

    @timer_log_method(logger_attr="logger")
    def load_measurement_data(self, data_type: str) -> pd.DataFrame:
        """Load measurement data of type Body or Bloodpressure"""
        from argus.data_loaders.factory import create_measurement_loader

        loader = create_measurement_loader(
            mode=self.mode,
            logger=self.logger,
            version=self.version,
            data_type=data_type,
            path_manager=self.path_manager,
            from_date=self.from_date,
        )
        return loader.load_data()

    @timer_log_method(logger_attr="logger")
    def process_measurements(self, df_body_measurements, df_BP_measurements) -> pd.DataFrame:
        measurement_processor = MetingenDataProcessor(
            df_body_measurements, df_BP_measurements, version=self.version, logger=self.logger, mode=self.mode
        )

        df_processed_metingen = measurement_processor.run_ETL_meting()
        self.logger.info("Metingen data processed successfully.")
        df_processed_metingen_bp = measurement_processor.run_ETL_meting_bp()
        self.logger.info("Metingen BP data processed successfully.")

        return df_processed_metingen, df_processed_metingen_bp

    @timer_log_method(logger_attr="logger")
    def main(self):
        df_body_measurements = self.load_measurement_data(data_type="body")
        df_BP_measurements = self.load_measurement_data(data_type="BP")
        df_processed_metingen, df_processed_metingen_bp = self.process_measurements(
            df_body_measurements, df_BP_measurements
        )

        return df_processed_metingen, df_processed_metingen_bp
