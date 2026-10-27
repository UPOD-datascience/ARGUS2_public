import datetime
from typing import Optional

from argus.utils.logger import Logger
from argus.utils.PathManager import PathManager_RES_Folder_UPOD

from .base import BaseDataLoader
from .inference import (
    InferenceDecursusLoader,
    InferenceECGLoader,
    InferenceLabLoader,
    InferenceMeasurementLoader,
    InferencePatselLoader,
)
from .training import (
    TrainingDecursusLoader,
    TrainingECGLoader,
    TrainingLabLoader,
    TrainingMeasurementLoader,
    TrainingPatselLoader,
)


def create_patsel_loader(
    mode: str,
    logger: Logger,
    version: str = "v2_ARGUS25",
    path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
    from_date: Optional[datetime.datetime] = None,
) -> BaseDataLoader:
    """Factory for patsel data loaders"""
    if mode == "train":
        if path_manager is None:
            raise ValueError("path_manager required for train mode")
        return TrainingPatselLoader(logger, version, path_manager)

    elif mode == "inference":
        # No validation needed - InferencePatselLoader will check and use .env if not provided
        return InferencePatselLoader(logger, version, db_user, db_password, from_date)

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'inference'")


def create_decursus_loader(
    mode: str,
    logger: Logger,
    version: str = "v2_ARGUS25",
    path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
    debug_mode: Optional[str] = False,
    from_date: Optional[datetime.datetime] = None,
) -> BaseDataLoader:
    """Factory for decursus data loaders"""
    if mode == "train":
        if path_manager is None:
            raise ValueError("path_manager required for train mode")
        return TrainingDecursusLoader(logger, version, path_manager)

    elif mode == "inference":
        # No validation needed - InferenceDecursusLoader will check and use .env if not provided
        return InferenceDecursusLoader(logger, version, db_user, db_password, debug_mode, from_date)

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'inference'")


def create_lab_loader(
    mode: str,
    logger: Logger,
    version: str = "v2_ARGUS25",
    path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
    from_date: Optional[datetime.datetime] = None,
) -> BaseDataLoader:
    """Factory for lab data loaders"""
    if mode == "train":
        if path_manager is None:
            raise ValueError("path_manager required for train mode")
        return TrainingLabLoader(logger, version, path_manager)

    elif mode == "inference":
        # No validation needed - InferenceLabLoader will check and use .env if not provided
        return InferenceLabLoader(logger, version, db_user, db_password, from_date)

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'inference'")


def create_ecg_loader(
    mode: str,
    logger: Logger,
    version: str = "v2_ARGUS25",
    ecg_type: str = "classresult",
    path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
    from_date: Optional[datetime.datetime] = None,
) -> BaseDataLoader:
    """Factory for ecg data loaders"""
    if mode == "train":
        if path_manager is None:
            raise ValueError("path_manager required for train mode")
        return TrainingECGLoader(logger, version, path_manager, ecg_type)

    elif mode == "inference":
        # No validation needed - InferenceECGLoader will check and use .env if not provided
        return InferenceECGLoader(logger, version, db_user, db_password, ecg_type, from_date)

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'inference'")


def create_measurement_loader(
    mode: str,
    logger: Logger,
    version: str = "v2_ARGUS25",
    data_type: str = "body",
    path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
    from_date: Optional[datetime.datetime] = None,
) -> BaseDataLoader:
    """Factory for measurement data loaders"""
    if mode == "train":
        if path_manager is None:
            raise ValueError("path_manager required for train mode")
        return TrainingMeasurementLoader(logger, version, path_manager, data_type)

    elif mode == "inference":
        # No validation needed - InferenceMeasurementLoader will check and use .env if not provided
        return InferenceMeasurementLoader(logger, version, db_user, db_password, data_type, from_date)

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'inference'")
