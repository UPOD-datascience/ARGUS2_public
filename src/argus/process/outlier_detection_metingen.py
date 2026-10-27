import logging

import numpy as np
import pandas as pd

from argus.utils.config import load_config


class MetingenOutlierDetector:
    """Class for detecting and handling outliers in medical measurements data.

    This class provides functionality to:
    - Detect outliers based on configurable ranges (in config_outliers.yaml)
    - Handle outliers by setting them to NaN
    - Log outlier information for tracking (in outliers.log)
    Version: 1.0.0
    """

    def __init__(self, config_file, version=None, logger=None, mode=None):
        """Initialize the MetingenOutlierDetector with configuration from YAML file."""
        # Set up logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Load configuration
        self.config = load_config(config_file)
        self.config_general = load_config("general", version=version, mode=mode)
        self.config_metingen = load_config("metingen", version=version, mode=mode)

        # Extract configuration values
        self.ranges = self.config.get("OUTLIER_RANGES", {})

        # Extract column names from configuration
        self.studyId = self.config_general["columns"]["study_id"]
        self.gender = self.config_general["columns"]["gender"]
        self.age = self.config_general["columns"]["age"]
        self.index_date = self.config_general["columns"]["index_date"]

        # Set metingen column names
        self.gewicht = self.config_metingen["columns"]["meting_gewicht"]
        self.lengte = self.config_metingen["columns"]["meting_lengte"]
        self.bmi = self.config_metingen["columns"]["meting_bmi"]
        self.bp_dia = self.config_metingen["columns"]["meting_bp_dia"]
        self.bp_syst = self.config_metingen["columns"]["meting_bp_syst"]
        self.bp_mean = self.config_metingen["columns"]["meting_bp_mean"]

        self.logger.info("MetingenOutlierDetector initialization complete.")

    def detect_outliers(self, df, column, min_val: float, max_val: float):
        """Detect outliers in a specific column based on min/max values."""
        if column not in df.columns:
            self.logger.warning(f"Column '{column}' not found in DataFrame")
            return pd.Series(dtype=bool, index=df.index)

        mask = (df[column] < min_val) | (df[column] > max_val)
        outlier_count = mask.sum()

        if outlier_count > 0:
            self.logger.info(f"Detected {outlier_count} outliers in '{column}' (outside range {min_val}-{max_val})")
        else:
            self.logger.info(f"No outliers detected in '{column}'")

        return mask

    def handle_outliers(
        self,
        df,
        column: str,
        min_val: float,
        max_val: float,
        id_col,
        date_col,
        extra_cols=None,
        replacement_value=np.nan,
    ):
        """Set outlier values to NaN and log the changes."""
        if extra_cols is None:
            extra_cols = []

        df_result = df.copy()

        # Find outliers
        mask = self.detect_outliers(df_result, column, min_val, max_val)

        # Define columns to log
        log_columns = [id_col, date_col, column] + extra_cols
        log_columns = [col for col in log_columns if col in df_result.columns]

        # Get the rows before modification
        log_df = df_result.loc[mask, log_columns].copy()

        if not log_df.empty:
            self.logger.info(f"{column}: Setting {log_df.shape[0]} outlier values to NaN (outside {min_val}-{max_val})")

            # Log outlier values for debugging
            if not log_df.empty:
                cols_to_log = [column] + extra_cols
                self.logger.debug(f"Outlier values in '{column}':\n{log_df[cols_to_log].to_string(index=False)}")

        # Set outliers to NaN
        df_result.loc[mask, column] = replacement_value

        return df_result

    def process_outliers_gw(self, df, id_col, date_col):
        df_result = df.copy()
        self.logger.info("Starting outlier detection and handling for Lengte and Gewicht configured columns")

        # Process Lengte and Gewicht:
        for column_name in [self.lengte, self.gewicht]:
            if column_name not in df_result.columns:
                self.logger.warning(f"Column '{column_name}' not found in DataFrame, skipping")
                continue

            range_config = self.ranges.get(column_name, {})
            min_val = range_config["min"]
            max_val = range_config["max"]
            extra_cols = range_config.get("extra_log_columns", [])

            # Filter extra_cols to only include existing columns
            extra_cols = [col for col in extra_cols if col in df_result.columns]

            self.logger.info(f"Processing outliers for '{column_name}'")

            df_result = self.handle_outliers(df_result, column_name, min_val, max_val, id_col, date_col, extra_cols)
        self.logger.info("Completed outlier detection and handling for Lengte and Gewicht columns")
        return df_result

    def calculate_bmi(self, df):
        """Calculate BMI after outlier handling so that outliers are also set to NaN."""
        height_col = self.lengte
        weight_col = self.gewicht
        bmi_col = self.bmi

        df_result = df.copy()

        # Check if required columns exist
        missing_cols = [col for col in [height_col, weight_col] if col not in df_result.columns]
        if missing_cols:
            self.logger.error(f"Missing columns for BMI calculation: {missing_cols}")
            return df_result

        # Calculate BMI (weight in kg / (height in cm / 100)^2)
        height_m = df_result[height_col] / 100  # Convert cm to meters

        # Handle cases where height_m is zero or NaN
        valid_height_mask = (height_m > 0) & (~height_m.isna())
        df_result.loc[~valid_height_mask, bmi_col] = np.nan  # Set BMI to NaN for invalid heights

        df_result.loc[valid_height_mask, bmi_col] = df_result.loc[valid_height_mask, weight_col] / (
            height_m[valid_height_mask] ** 2
        )

        # round the BMI to 1 decimal
        df_result[bmi_col] = df_result[bmi_col].round(1)

        return df_result

    def handle_outliers_bp(self, df):
        """Handle outliers for blood pressure measurements."""
        self.logger.info("Starting outlier detection and handling for BP measurements")
        df_result = df.copy()

        # get the range for BP measurements from config
        range_config_syst = self.ranges.get(self.bp_syst, {})
        sysBP_min = range_config_syst["min"]
        sysBP_max = range_config_syst["max"]
        extra_cols_syst = range_config_syst.get("extra_log_columns", [])

        range_config_dia = self.ranges.get(self.bp_dia, {})
        diaBP_min = range_config_dia["min"]
        diaBP_max = range_config_dia["max"]
        extra_cols_dia = range_config_dia.get("extra_log_columns", [])

        # If diastolic BP is less then diaBP_min, set to NaN:
        low_dia_mask = df_result[self.bp_dia] < diaBP_min
        if low_dia_mask.any():
            self.logger.info(f"Setting {low_dia_mask.sum()} diastolic BP values below {diaBP_min} to NaN")
            # Log the outlier values
            log_columns = [self.studyId, self.index_date, self.bp_dia] + extra_cols_dia
            log_df = df_result.loc[low_dia_mask, log_columns].copy()
            if not log_df.empty:
                self.logger.debug(f"Outlier diastolic BP values:\n{log_df.to_string(index=False)}")
            # Set outliers to NaN
            df_result.loc[low_dia_mask, self.bp_dia] = np.nan

        # Check for systolic values that are suspiciously large (5+ digits) and could be concatenated BP values
        concat_mask = (
            (df_result[self.bp_syst] >= 10000) & (df_result[self.bp_syst] < 30000) & (df_result[self.bp_dia].isna())
        )

        if concat_mask.any():
            self.logger.info(f"Detected {concat_mask.sum()} concatenated BP values in systolic BP column")
            # Log the outlier values
            log_columns = [self.studyId, self.index_date, self.bp_syst] + extra_cols_syst
            log_df = df_result.loc[concat_mask, log_columns].copy()
            if not log_df.empty:
                self.logger.debug(f"Concatenated systolic BP values:\n{log_df.to_string(index=False)}")

        # Process concatenated BP values using vectorized operations
        combined_values = df_result.loc[concat_mask, self.bp_syst].astype(int).astype(str)
        sys_values = combined_values.str[:-2].astype(int)
        dia_values = combined_values.str[-2:].astype(int)

        valid_mask = (
            (sysBP_min <= sys_values)
            & (sys_values <= sysBP_max)
            & (diaBP_min <= dia_values)
            & (dia_values <= diaBP_max)
            & (dia_values < sys_values)  # Ensure diastolic is less than systolic
        )

        df_result.loc[concat_mask & valid_mask, self.bp_syst] = sys_values[valid_mask]
        df_result.loc[concat_mask & valid_mask, self.bp_dia] = dia_values[valid_mask]

        self.logger.info(f"Fixed {valid_mask.sum()} concatenated BP values using vectorized operations")

        # Now process the regular out-of-range values using handle_outliers
        # Process diastolic BP outliers
        df_result = self.handle_outliers(
            df_result, self.bp_dia, diaBP_min, diaBP_max, self.studyId, self.index_date, extra_cols_dia
        )
        # Process systolic BP outliers
        df_result = self.handle_outliers(
            df_result, self.bp_syst, sysBP_min, sysBP_max, self.studyId, self.index_date, extra_cols_syst
        )

        # Next, check the values if DiastolicBP is greater than SystolicBP:
        mask = df_result[self.bp_dia] > df_result[self.bp_syst]
        if mask.any():
            self.logger.info(f"\nFound {mask.sum()} rows where DiastolicBP > SystolicBP:")
            # Log the outlier values
            log_columns = [self.studyId, self.index_date, self.bp_syst, self.bp_dia] + extra_cols_syst + extra_cols_dia
            log_df = df_result.loc[mask, log_columns].copy()
            if not log_df.empty:
                self.logger.debug(f"Outlier BP values (Diastolic > Systolic):\n{log_df.to_string(index=False)}")

        # Lastly, for each row where DiastolicBP or SystolicBP is NaN, set MeanBP to NaN:
        df_result.loc[df_result[self.bp_dia].isna() | df_result[self.bp_syst].isna(), self.bp_mean] = np.nan

        self.logger.info("Completed outlier detection and handling for BP measurements")
        return df_result

    def run_pipeline(self, df):
        """Run the complete outlier detection and BMI calculation pipeline."""
        self.logger.info("Starting complete outlier detection and BMI calculation pipeline")

        # Process outliers Gewicht and Lengte
        df_processed = self.process_outliers_gw(df, self.studyId, self.index_date)
        self.logger.info("Calculating BMI after outlier handling")
        df_processed = self.calculate_bmi(df_processed)

        # Handle outliers for BP measurements
        df_processed_bp = self.handle_outliers_bp(df_processed)
        self.logger.info("Outlier detection and handling for BP measurements complete")

        # Keep only nesecary cols
        expected_cols = [
            self.studyId,
            self.index_date,
            self.gender,
            self.age,
            self.gewicht,
            self.lengte,
            self.bmi,
            self.bp_dia,
            self.bp_syst,
            self.bp_mean,
        ]

        missing = [c for c in expected_cols if c not in df_processed_bp.columns]
        if missing:
            self.logger.warning(f"Missing expected columns in pipeline output, adding as NaN: {missing}")

        # reindex will keep column order and insert missing columns filled with NaN
        df_processed_bp = df_processed_bp.reindex(columns=expected_cols)

        self.logger.info("Pipeline processing complete")
        return df_processed_bp
