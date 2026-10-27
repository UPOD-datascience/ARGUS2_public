import pandas as pd

from argus.utils.config import load_config


class DataProcessingUtils:
    """General functions that are used more often on several parts of the ARGUS dataset."""

    @staticmethod
    def validate_input_data(df, required_cols, id_col, logger=None):
        """Validate that input data is a DataFrame with required columns.

        Args:
            df (pd.DataFrame): DataFrame to validate
            required_cols (list): List of column names that must be in the DataFrame
            id_col (str): Column name used as the record identifier to validate
            logger: Logger object with error method for logging

        Returns:
            bool: True if validation passes, raises exceptions otherwise

        """
        try:
            if not isinstance(df, pd.DataFrame):
                if logger:
                    logger.error("Input data is not a DataFrame.")
                raise TypeError("Input data must be a pandas DataFrame.")

            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                if logger:
                    logger.error(f"Missing columns in input data: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            if not df[id_col].dtype == "object":
                # covert column to string
                df[id_col] = df[id_col].astype(int)  # Remove floats
                df[id_col] = df[id_col].astype(str)
                if logger:
                    logger.debug(f"Converted {id_col} to string type.")

            return True

        except Exception as e:
            if logger:
                logger.error(f"Error validating input data: {str(e)}")
            raise

    @staticmethod
    def map_gender_values(df, config=None, logger=None):
        """Map gender text values to numeric codes. Male = 0 and Female = 1.

        Args:
            df (pd.DataFrame): DataFrame with a 'gender' column
            config: Name of the config file type (e.g. general, ecg or lab)
            logger: Logger object with warning method for logging

        Returns:
            pd.DataFrame: DataFrame with mapped gender values

        """
        try:
            if config is None:
                config = load_config("general")

            # Get column names and mapping from config
            gender_col = config["columns"]["gender"]
            study_id_col = config["columns"]["study_id"]
            gender_map = config["mappings"]["gender"]

            # Create a copy to avoid modifying the original
            df_copy = df.copy()

            # Store original gender values before mapping
            original_gender = df_copy[gender_col].copy()

            # Map gender values
            df_copy.loc[:, gender_col] = df_copy[gender_col].map(gender_map)

            # Check for unmapped values
            unmapped_mask = df_copy[gender_col].isna()
            if unmapped_mask.any() and logger and study_id_col in df_copy.columns:
                # Create a list of studyId and original_gender_value
                unmapped_records = list(
                    zip(
                        df_copy.loc[unmapped_mask, study_id_col].tolist(),
                        original_gender.loc[unmapped_mask].tolist(),
                        strict=False,
                    )
                )

                # Log with both study IDs and original gender values
                logger.warning(
                    f"Warning: Found {len(unmapped_records)} records with unmapped gender values: {unmapped_records[:5]}"  # noqa: E501
                )

            return df_copy

        except Exception as e:
            if logger:
                logger.error(f"Error mapping gender values: {str(e)}")
            raise

    @staticmethod
    def handle_missing_data(df, data_columns, id_col=None, config=None, logger=None):
        """Handle missing drop rows where values are missing in the specified columns."""
        # Load config if not provided
        if config is None:
            config = load_config()

        # Get column names and settings from config
        if id_col is None:
            id_col = config["columns"]["study_id"]

        for data_column in data_columns:
            if df[data_column].isnull().any():
                # Identify rows with missing values
                rows_to_drop = df[df[data_column].isnull()]

                # Log the count of rows and the corresponding study_id values
                logger.warning(f"Missing values in {data_column}. Rows deleted: {len(rows_to_drop)}")
                logger.warning(f"studyId for deleted rows: {', '.join(map(str, rows_to_drop[id_col].values))}")

                # Drop rows with missing values
                df = df.dropna(subset=[data_column])
        return df

    @staticmethod
    def convert_date_columns(df, date_cols, config=None, id_col=None, logger=None):
        """Convert string date columns to datetime objects.

        Args:
            df (pd.DataFrame): DataFrame with date columns
            date_cols (list): List of column names to convert to datetime
            config (dict): Configuration dictionary (if None, will load default)
            id_col (str): Column name for ID field used in logging
            logger: Logger object with debug method for logging

        Returns:
            pd.DataFrame: DataFrame with converted date columns

        """
        try:
            if config is None:
                config = load_config()

            if id_col is None:
                id_col = config["columns"]["study_id"]

            index_date_col = config["columns"]["index_date"]  # CT_Date or index_date

            df_copy = df.copy()

            for col in date_cols:
                if col in df_copy.columns:
                    original_values = df_copy[col].copy()

                    # Special handling for index_date or CT_Date (columns which should always be datetime)
                    if col == index_date_col and logger:
                        # Find rows that were NaN before conversion
                        pre_conversion_nan_mask = original_values.isna()
                        pre_conversion_nan_ids = []

                        if pre_conversion_nan_mask.any():
                            pre_conversion_nan_ids = df_copy.loc[pre_conversion_nan_mask, id_col].tolist()
                            logger.debug(
                                f"Debug: {len(pre_conversion_nan_ids)} records had no {col} values before conversion:\
                                     {pre_conversion_nan_ids}"
                            )

                        # Convert to datetime
                        df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")

                        # Find rows where conversion failed (were not NaN before but are NaT now)
                        post_conversion_nan_mask = df_copy[col].isna()
                        failed_conversion_mask = post_conversion_nan_mask & ~pre_conversion_nan_mask

                        if failed_conversion_mask.any():
                            # Get the study IDs and original values that failed conversion
                            failed_records = list(
                                zip(
                                    df_copy.loc[failed_conversion_mask, id_col].tolist(),
                                    original_values.loc[failed_conversion_mask].tolist(),
                                    strict=False,
                                )
                            )

                            # Log with study IDs and their original values
                            logger.debug(
                                f"Debug: {len(failed_records)} {col} values could not be converted to dates:\
                                     {failed_records}",
                            )

                        # Find rows that became NaN after conversion
                        # includes both pre-conversion NaNs and failed conversions
                        post_conversion_nan_ids = df_copy.loc[post_conversion_nan_mask, id_col].tolist()
                        if post_conversion_nan_ids:
                            logger.debug(
                                f"Debug: After conversion, {len(post_conversion_nan_ids)} records have NaN {col}\
                                values: {post_conversion_nan_ids}"
                            )

                    else:
                        # Convert to datetime without logging
                        df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")

            return df_copy

        except Exception as e:
            if logger:
                logger.error(f"Error converting date columns: {str(e)}")
            raise

    @staticmethod
    def create_time_features(df, time_col, logger=None):
        """Create hour and date features from a datetime column."""
        try:
            if time_col not in df.columns:
                raise ValueError(f"Column {time_col} not found in DataFrame.")

            # Convert to datetime if not already
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

            # Set column names
            date_col = f"{time_col}_date"
            hour_col = f"{time_col}_time"

            # Create hour and date features
            df = df.copy()
            df = df.assign(**{hour_col: df[time_col].dt.time})
            df = df.assign(**{date_col: df[time_col].dt.date})
            return df

        except Exception as e:
            if logger:
                logger.error(f"Error creating time features: {str(e)}")
            raise
