import logging
import time

import pandas as pd

from argus.utils.config import load_config
from argus.utils.general_argus_functions import DataProcessingUtils


class LabDataProcessor:
    """Class for preprocessing and merging lab data with patient data.

    Attributes:
        dfLab (pd.DataFrame): DataFrame containing lab test data.
        lab_features (list): List of lab features (codes) to process.

    """

    def __init__(self, dfLab, logger=None, version=None, mode=None):
        """Initializes the LabDataProcessor with dataframes and configuration parameters."""
        self.dfLab = dfLab.copy()
        self.version = version
        self.mode = mode

        # Set up logger
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

        # Load configurations
        self.config = load_config("lab")
        self.config_general = load_config("general", version=version, mode=mode)

        # Set general column names
        self.studyId = self.config_general["columns"]["study_id"]
        self.gender = self.config_general["columns"]["gender"]
        self.index_date = self.config_general["columns"]["index_date"]
        self.age = self.config_general["columns"]["age"]

        # Set lab specific column names
        self.lab_code = self.config["etl"]["columns"]["lab_code"]
        self.lab_value = self.config["etl"]["columns"]["lab_value"]
        self.lab_test_date = self.config["etl"]["columns"]["lab_test_date"]
        self.lab_variable = self.config["etl"]["columns"]["lab_variable"]
        self.lab_variable_description = self.config["etl"]["columns"]["lab_variable_description"]

        # Set lab parameters
        self.lab_code_list = self.config["etl"]["lab"]["lab_code_list"]
        self.lab_features = self.config["etl"]["lab"]["lab_features"]
        self.alpha = self.config["etl"]["lab"]["alpha"]
        self.days_after = self.config["etl"]["lab"]["days_after"]
        self.days_prior = self.config["etl"]["lab"]["days_prior"]

    def preprocess_lab(self) -> pd.DataFrame:
        """Preprocess lab dataframe by filtering rows and selecting relevant columns"""
        try:
            self.log.info("Starting lab data preprocessing.")

            # Step 1: Validate input
            required_cols = [
                self.studyId,
                self.lab_code,
                self.lab_value,
                self.lab_test_date,
            ]
            DataProcessingUtils.validate_input_data(self.dfLab, required_cols, id_col=self.studyId, logger=self.log)

            # Step 2: Drop all rows with NaN in 'valueQuantity_value'
            DataProcessingUtils.handle_missing_data(self.dfLab, [self.lab_value], id_col=self.studyId, logger=self.log)

            # Step 3: Filter lab data for specific features
            self.dfLab = self.dfLab.loc[
                self.dfLab[self.lab_code].isin(self.lab_code_list),
                [
                    self.studyId,
                    self.lab_code,
                    self.lab_variable,
                    self.lab_variable_description,
                    self.lab_value,
                    self.lab_test_date,
                ],
            ]
            self.dfLab = self.dfLab.reset_index(drop=True)

            # Step 4: Convert 'effectiveDateTime' to datetime
            self.dfLab[self.lab_test_date] = pd.to_datetime(self.dfLab[self.lab_test_date], errors="coerce")

            self.log.info("Successfully preprocessed lab data.")
            return self.dfLab

        except Exception as e:
            self.log.error(f"Error preprocessing lab data: {str(e)}")
            raise

    def split_lab_by_feature(self, lab: pd.DataFrame) -> dict:
        """Splits the lab data by features and stores each feature in a separate DataFrame.

        Returns: dictionary with feature names as keys and DataFrames as values.
        """
        self.log.info("Splitting lab data by features.")
        lab_dfs = {}
        for feature in self.lab_features:
            lab_dfs[feature] = (
                lab.loc[
                    lab[self.lab_variable] == feature,
                    [
                        self.studyId,
                        self.lab_test_date,
                        self.lab_value,
                    ],
                ]
                .rename(columns={self.lab_value: feature})
                .reset_index(drop=True)
            )

        return lab_dfs

    def combine_lab_features(self, lab_dfs: dict) -> pd.DataFrame:
        """Combines all lab feature DataFrames into a single DataFrame.

        The resulting DataFrame will have studyId and lab_test_date as the key columns,
        with one column per lab feature. Missing values will be filled with NaN.
        """
        self.log.info("Combining lab features into DataFrame.")

        # Create a single DataFrame with an identifier column for the feature
        dfs_with_feature_col = []

        for feature, df in lab_dfs.items():
            # Create a copy with just the columns we need
            temp_df = df[[self.studyId, self.lab_test_date, feature]].copy()

            # Rename the feature column to a generic name and add feature identifier
            temp_df = temp_df.rename(columns={feature: "value"})
            temp_df["feature"] = feature
            dfs_with_feature_col.append(temp_df)

        # Concatenate all DataFrames into one
        all_data = pd.concat(dfs_with_feature_col, ignore_index=True)

        # Find duplicates lab values for the same studyId and date
        key_columns = [self.studyId, self.lab_test_date, "feature"]

        # Check if the lab values differ if the same patients has multiple lab tests on
        # the same date. Note: if they are the same it can be ignored.
        value_differences = all_data.groupby(key_columns)["value"].nunique() > 1
        has_different_values = value_differences[value_differences].index.tolist()

        if has_different_values:
            different_values_count = len(has_different_values)
            self.log.warning(
                f"Found {different_values_count} instances where the same feature has"
                "different values for the same studyId and date."
            )

            # Count differences by feature type
            feature_counts = {}
            for _, _lab_date, feature in has_different_values:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

            self.log.warning(f"Counts of value differences by feature: {feature_counts}")

            # Log examples of the differences (limit to first 5)
            sample_count = min(5, different_values_count)
            for i in range(sample_count):
                study_id, lab_date, feature = has_different_values[i]
                matching_rows = all_data[
                    (all_data[self.studyId] == study_id)
                    & (all_data[self.lab_test_date] == lab_date)
                    & (all_data["feature"] == feature)
                ]
                values = matching_rows["value"].tolist()
                self.log.warning(f"Different values for {study_id}, {lab_date}, {feature}: {values}")

            # Log ALL examples of the differences to debug
            self.log.debug(f"Logging all {different_values_count} instances of different values:")

            for idx, (study_id, lab_date, feature) in enumerate(has_different_values):
                matching_rows = all_data[
                    (all_data[self.studyId] == study_id)
                    & (all_data[self.lab_test_date] == lab_date)
                    & (all_data["feature"] == feature)
                ]
                values = matching_rows["value"].tolist()
                mean_value = matching_rows["value"].mean()
                self.log.debug(
                    f"{idx + 1}. Different values for {study_id}, {lab_date},"
                    f"{feature}: {values} - using mean: {mean_value}"
                )

        # Pivot the data to get features as columns
        pivoted = all_data.pivot_table(
            index=[self.studyId, self.lab_test_date],
            columns="feature",
            values="value",
            aggfunc="mean",  # In case there are duplicates
        )

        # Round all values to 1 decimal
        pivoted = pivoted.round(1)
        pivoted.reset_index(inplace=True)

        self.log.info(f"Pivoted DataFrame created with {len(pivoted)} rows and{len(pivoted.columns)} columns.")
        self.log.info(f"Pivoted DataFrame created with {pivoted[self.studyId].nunique()} patients")
        return pivoted

    def process_all_features(self) -> pd.DataFrame:
        """Processes all features by splitting the lab data and merging with the
        patient data.

        Returns: A merged DataFrame containing processed lab data for all features.
        """
        try:
            # Start timer for the whole pipeline
            start_time = time.time()
            self.log.info("Starting lab feature processing pipeline.")

            # Step 1: Preprocess lab data
            step_start = time.time()
            self.lab = self.preprocess_lab()
            step_duration = time.time() - step_start
            self.log.info(f"Lab data preprocessed. Time taken: {step_duration:.2f} seconds.")

            # Step 2: Split lab data by feature
            step_start = time.time()
            lab_dfs = self.split_lab_by_feature(self.lab)
            step_duration = time.time() - step_start
            self.log.info(f"Lab data split by feature. Time taken: {step_duration:.2f} seconds.")

            # Step 3: Combine lab features into a single DataFrame
            step_start = time.time()
            combined_lab_df = self.combine_lab_features(lab_dfs)
            step_duration = time.time() - step_start
            self.log.info(f"Lab features combined into a single DataFrame. Time taken:{step_duration:.2f} seconds.")

            # Total time for the whole pipeline
            total_duration = time.time() - start_time
            self.log.info(
                "Lab feature processing pipeline completed successfully."
                f"Total time taken: {total_duration:.2f} seconds."
            )

            return combined_lab_df

        except Exception as e:
            self.log.error(f"Error in processing lab features: {str(e)}")
            raise
