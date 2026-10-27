import logging

import numpy as np

from argus.utils.config import load_config
from argus.utils.general_argus_functions import DataProcessingUtils


class MetingenDataProcessor:
    ### Note: for the new dataset we have to check if index date is present and
    # if column names are correct. ###
    """Initializes the MetingenDataProcessor class.
    This class is responsible for loading, preprocessing, and merging metingen
    data with patient selection data.
    """

    def __init__(self, dfMeting, dfMeting_bp=None, version=None, logger=None, mode=None):
        self.df_meting = dfMeting.copy()
        self.version = version
        self.mode = mode

        if dfMeting_bp is not None:
            self.df_meting_bp = dfMeting_bp.copy()

        # Load configurations
        self.config = load_config("metingen", version=version, mode=mode)
        self.config_general = load_config("general", version=version, mode=mode)

        # Set general column names
        self.studyId = self.config_general["columns"]["study_id"]
        self.gender = self.config_general["columns"]["gender"]
        self.index_date = self.config_general["columns"]["index_date"]

        # Set metingen column names
        self.variable = self.config["columns"]["meting_variable_description"]
        self.test_date = self.config["columns"]["meting_test_date"]
        self.value = self.config["columns"]["meting_value"]
        self.gewicht = self.config["columns"]["meting_gewicht"]
        self.lengte = self.config["columns"]["meting_lengte"]
        self.bmi = self.config["columns"]["meting_bmi"]
        self.bp_dia = self.config["columns"]["meting_bp_dia"]
        self.bp_syst = self.config["columns"]["meting_bp_syst"]
        self.bp_mean = self.config["columns"]["meting_bp_mean"]
        self.bp_method = self.config["columns"]["meting_bp_method"]

        # Set up logger
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

    def set_id_col_type(self, df):
        """Set the ID column type to string."""
        # Validate studyId is string type
        if self.studyId in df.columns:
            if df[self.studyId].dtype != "object":
                # Convert to int first to remove floats
                df[self.studyId] = df[self.studyId].astype(int)  # Remove floats
                df[self.studyId] = df[self.studyId].astype(str)
        return df

    def preprocess_data(self):
        """Preprocesses the metingen data."""
        self.log.info("Starting data preprocessing.")

        # Validate input data
        required_columns = [self.studyId]
        DataProcessingUtils.validate_input_data(self.df_meting, required_columns, id_col=self.studyId, logger=self.log)

        # Filter rows
        self.df_meting = self.df_meting.loc[
            self.df_meting[self.variable].isin([self.gewicht, self.lengte]),
            [self.studyId, self.test_date, self.variable, self.value],
        ]
        self.log.info(f"Data filtered: {self.df_meting.shape[0]} rows.")

        # Drop rows with missing 'valueQuantity_value'
        self.df_meting = DataProcessingUtils.handle_missing_data(
            self.df_meting, [self.value], id_col=self.studyId, logger=self.log
        )

        # Pivot table to get Gewicht and Lengte in separate columns
        df_pivoted = self.df_meting.pivot_table(
            index=[self.studyId, self.test_date],
            columns=self.variable,
            values=[self.value],
            aggfunc="mean",
        )

        # Flatten column names
        df_pivoted.columns = ["_".join(col).strip() for col in df_pivoted.columns.values]  # noqa: E501
        df_pivoted = df_pivoted.reset_index()

        # # Rename columns for clarity
        df_pivoted.rename(
            columns={
                f"{self.value}_{self.gewicht}": self.gewicht,
                f"{self.value}_{self.lengte}": self.lengte,
            },
            inplace=True,
        )

        # Replace 0 values with NaN
        df_pivoted[self.gewicht] = df_pivoted[self.gewicht].astype(float).replace(0, np.nan)
        df_pivoted[self.lengte] = df_pivoted[self.lengte].astype(float).replace(0, np.nan)

        # # Calculate BMI
        df_pivoted[self.bmi] = (df_pivoted[self.gewicht] / ((df_pivoted[self.lengte] / 100) ** 2)).round(1)

        df_pivoted = df_pivoted.dropna(subset=["BMI"])
        return df_pivoted

    def filter_dataframe(self):
        """Filters Bloodpressure columns."""
        if self.df_meting_bp is not None:
            self.log.info("Filtering Bloodpressure data.")
            # Validate input data
            required_columns = [self.studyId]
            DataProcessingUtils.validate_input_data(
                self.df_meting_bp,
                required_columns,
                id_col=self.studyId,
                logger=self.log,
            )
            self.df_meting_bp = self.df_meting_bp[
                [
                    self.studyId,
                    self.test_date,
                    self.bp_dia,
                    self.bp_syst,
                    self.bp_mean,
                    self.bp_method,
                ]
            ].copy()
            return self.df_meting_bp
        else:
            self.log.error("Blood pressure data is not available.")
            raise ValueError("Blood pressure data is not available.")

    def run_ETL_meting(self):
        """Runs the entire meting data processing pipeline."""
        try:
            self.log.info("Starting the entire data processing pipeline.")
            # Set ID column type
            self.df_meting = self.set_id_col_type(self.df_meting)
            # Preprocess data
            df = self.preprocess_data()
            self.log.info("Pipeline completed successfully.")
            return df
        except Exception as e:
            self.log.error(f"Error during the pipeline execution: {e}")
            raise

    def run_ETL_meting_bp(self):
        """Runs the MetingenBP data processing pipeline."""
        try:
            self.log.info("Starting the MetingenBP data processing pipeline.")
            # Set ID column type
            self.df_meting_bp = self.set_id_col_type(self.df_meting_bp)
            # Filter and process Bloodpressure data
            filtered_df = self.filter_dataframe()
            self.log.info("Pipeline completed successfully.")
            return filtered_df
        except Exception as e:
            self.log.error(f"Error during MetingenBP processing pipeline: {e}")
            raise
