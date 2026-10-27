"""Utilities for creating a structured logging and DataFrame-backed log capture."""

import logging
import os
import traceback
from datetime import datetime

import pandas as pd


class DataFrameHandler(logging.Handler):
    """Capture log records and expose them as a pandas DataFrame."""

    def __init__(self):
        """Initialize a handler that stores log records for later inspection."""
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        """Capture a log record and convert it into a dictionary entry."""
        exc_text = None
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info))

        self.records.append(
            {
                "timestamp": datetime.fromtimestamp(record.created),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "file": os.path.basename(record.pathname) if record.pathname else None,
                "line": record.lineno,
                "module": record.module,
                "func": record.funcName,
                "exception": exc_text,  # <-- NEW (None unless exception)
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return all captured records as a pandas DataFrame."""
        return pd.DataFrame(self.records)


class Logger:
    """Logging class to be used in other classes."""

    def __init__(
        self,
        name=None,
        log_file=None,
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        df_level=logging.DEBUG,
        enabled=True,
    ):
        """Initialize logger.

        Args:
            name (str, optional): Logger name. If None, uses the module logger.
            log_file (str, optional): Path to the log file. If None, only
                console logging is used.
            console_level (int): Logging level for console output.
            file_level (int): Logging level for file output.
            df_level (int): Logging level for the DataFrame capture handler.
            enabled (bool): Whether logging is enabled.

        """
        self.enabled = enabled
        self.df_handler = None

        if not self.enabled:
            return

        # Create logger
        self.logger = logging.getLogger(name or __name__)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        self.logger.propagate = False  # prevent duplicates via root logger

        # Clear existing handlers to avoid duplication
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create file handler if log file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # DataFrame handler
        self.df_handler = DataFrameHandler()
        self.df_handler.setLevel(df_level)
        self.logger.addHandler(self.df_handler)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the collected dataframe log entries."""
        if not self.enabled or self.df_handler is None:
            return pd.DataFrame()
        return self.df_handler.to_dataframe()

    # Use stacklevel so record.pathname/lineno point to the *caller* of Logger.<method>()
    def info(self, message):
        """Log an informational message."""
        if self.enabled:
            self.logger.info(message, stacklevel=2)

    def warning(self, message):
        """Log a warning message."""
        if self.enabled:
            self.logger.warning(message, stacklevel=2)

    def error(self, message):
        """Log an error message."""
        if self.enabled:
            self.logger.error(message, stacklevel=2)

    def debug(self, message):
        """Log a debug message."""
        if self.enabled:
            self.logger.debug(message, stacklevel=2)

    def critical(self, message):
        """Log a critical message."""
        if self.enabled:
            self.logger.critical(message, stacklevel=2)

    def exception(self, message):
        """Log an exception with full traceback."""
        if self.enabled:
            self.logger.exception(message, stacklevel=2)
