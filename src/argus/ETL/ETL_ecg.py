import logging
import time

import numpy as np
import pandas as pd

from argus.utils.config import load_config
from argus.utils.general_argus_functions import DataProcessingUtils


class ECGdataProcessor:
    """Class for extracting, transforming and loading ECG data.

    This class handles the complete ECG data processing pipeline including:
    1. Data validation and preparation
    2. Lead transformation and pivoting
    3. Replace zeros with NaN (if set True)
    4. Region-based aggregation

    Attributes:
        nog in te vulllen

    """

    def __init__(self, dfECG, dfECG2=None, logger=None, version=None, mode=None):
        # Define pre-loaded dataframes
        self.dfECG = dfECG
        self.dfECG2 = dfECG2 if dfECG2 is not None else None

        # Load configurations
        self.mode = mode
        self.config = load_config("ecg", version=version, mode=mode)
        self.config_general = load_config("general", version=version, mode=mode)

        # Set column names
        self._set_column_names()

        # Set ECG parameters
        self._set_ecg_parameters()

        # Set metadata columns
        self.meta_cols = [
            self.studyId,
            self.ecg_indexdate,
            self.acquisitiondate,
            self.ecgtestId,
            self.leadIdText,
        ]

        # Set up logger
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

    def _set_column_names(self):
        self.studyId = self.config_general["columns"]["study_id"]
        self.gender = self.config_general["columns"]["gender"]
        self.index_date = self.config_general["columns"]["index_date"]
        self.age = self.config_general["columns"]["age"]

        self.acquisitiondate = self.config["columns_ecg"]["acquisition_date"]
        self.bitflag = self.config["columns_ecg"]["bitflag"]
        self.leadIdText = self.config["columns_ecg"]["leadIdText"]
        self.ecgtestId = self.config["columns_ecg"]["ecgtestId"]
        self.ecg_indexdate = self.config["columns_ecg"]["index_date"]
        self.incjclsinfarction = self.config["columns_ecg"]["InjClsInfarction"]

    def _set_ecg_parameters(self):
        self.days_after = self.config["etl"]["ecg"]["days_after"]
        self.days_prior = self.config["etl"]["ecg"]["days_prior"]
        self.varying_vars = self.config["etl"]["ecg"]["varying_vars"]
        self.lead_regions = self.config["etl"]["ecg"]["lead_regions"]
        self.replace_zeros_with_nan = self.config["etl"]["ecg"]["replace_zeros_with_nan"]

    def _handle_missing_data(self, df, data_columns):
        """Handle missing drop rows where values are missing in the specified columns."""  # noqa: E501
        for data_column in data_columns:
            if df[data_column].isnull().any():
                # Identify rows with missing values
                rows_to_drop = df[df[data_column].isnull()]

                # Log the count of rows and the corresponding study_id values
                self.log.warning(
                    f"Missing values in {data_column}. Rows deleted: {len(rows_to_drop)}"  # noqa: E501
                )
                message = rows_to_drop[self.studyId].values
                self.log.warning(f"studyId for deleted rows: {', '.join(map(str, message))}")

                # Drop rows with missing values
                df = df.dropna(subset=[data_column])
        return df

    def validate_and_prepare_ecg_data(self):
        """Validates ECG data to ensure it meets requirements for processing.

        This function handles validation of input data structure, required columns,
        and handling of missing or invalid values.
        """
        try:
            self.log.info("ETL ECG data started.")
            df_ecg = self.dfECG

            # Validate input data
            required_columns = [
                self.acquisitiondate,
                self.leadIdText,
            ] + self.varying_vars
            DataProcessingUtils.validate_input_data(df_ecg, required_columns, id_col=self.studyId, logger=self.log)

            # Drop rows with missing values in ecg date
            df_ecg = self._handle_missing_data(df_ecg, [self.acquisitiondate])
            self.log.info("ECG data processed successfully.")
            return df_ecg

        except Exception as e:
            self.log.error(f"Error processing ECG data: {str(e)}")
            raise

    def transform_ecg_leads(self, df_ecg):
        """Transforms ECG data from raw format to analysis-ready format.

        This function handles pivoting, feature engineering, and data transformation
        for ECG data, including region-based aggregation of measurements.
        """
        try:
            self.log.info("Transform ECG data by pivoting leads into columns.")

            # Check if 'meta_cols' are provided and valid
            if (
                not hasattr(self, "meta_cols")
                or not isinstance(self.meta_cols, list)
                or not all(isinstance(col, str) for col in self.meta_cols)
            ):
                self.log.error("'meta_cols' is not defined or is not a valid list of strings.")
                raise ValueError("'meta_cols' should be a list of strings.")

            # Define index columns (all meta columns except the lead column)
            index_cols = [c for c in self.meta_cols if c != self.leadIdText]

            # Pivot the table to create dynamic variables
            df_pivoted = df_ecg.pivot_table(index=index_cols, columns=self.leadIdText, values=self.varying_vars)

            self.log.info(f"Pivot table created. Shape: {df_pivoted.shape}")

            # Flatten the multi-level column index
            df_pivoted.columns = ["_".join(col).strip() for col in df_pivoted.columns.values]

            # Prefix the column names
            df_pivoted.columns = ["ECG_" + col for col in df_pivoted.columns]

            self.log.info(f"Pivot table flattened. Shape: {df_pivoted.shape}")

            df_pivoted = df_pivoted.drop_duplicates()
            self.log.info(f"Pivot table duplicate rows dropped. Shape: {df_pivoted.shape}")

            self.log.info("ECG data transformation completed successfully.")

            return df_pivoted

        except Exception as e:
            self.log.error(f"Error during ECG lead transformation: {str(e)}")
            raise

    def _process_zeros(self, df):
        """Process zeros in the data, replacing with NaN if configured."""
        try:
            df_result = df.copy()
            if self.replace_zeros_with_nan:
                df_result.replace(0, np.nan, inplace=True)
                message = df_result.isna().sum().sum()
                self.log.info(f"Replaced zeros with NaN. Cells replaced: {message}")
            return df_result

        except Exception as e:
            self.log.error(f"Error replacing zeros with NaN: {str(e)}")
            raise

    def _aggregate_by_region(self, df, feature_prefix):
        """Aggregates ECG features by region."""
        region_features = {}  # Store aggregated features

        try:
            for region in set(self.lead_regions.values()):
                region_cols = [
                    col
                    for col in df.columns
                    if feature_prefix in col
                    and any(lead in col for lead, reg in self.lead_regions.items() if reg == region)
                ]

                if region_cols:
                    # Aggregate using mean (or max/min depending on clinical relevance)
                    region_features[f"{feature_prefix}_{region}"] = df[region_cols].mean(axis=1)

            return pd.DataFrame(region_features)

        except Exception as e:
            self.log.error(f"Error during aggregation by region: {str(e)}")
            raise

    def group_ecg_by_region(self, df_ecg, cad_features):
        """Groups ECG data by region and aggregates features."""
        # Aggregate all features by region

        self.log.info("Start Aggregating ECG features by region.")
        region_based_features = pd.concat(
            [self._aggregate_by_region(df_ecg, feature) for feature in cad_features],
            axis=1,
        )
        self.log.info("Aggregated features by region successfully.")

        ECG_region = region_based_features.copy()

        return ECG_region

    def merge_ecg_detail_classresults(self, df_detail, df_classresults):
        # Check if naming for index_dt is the same for all datasets (needed for merging)
        if self.ecg_indexdate in df_detail.columns and self.ecg_indexdate != self.index_date:
            df_detail = df_detail.rename(columns={self.ecg_indexdate: self.index_date})
            self.log.info(f"ECG_detail - Renamed column {self.ecg_indexdate} to {self.index_date}")

        if self.ecg_indexdate in df_classresults.columns and self.ecg_indexdate != self.index_date:
            df_classresults = df_classresults.rename(columns={self.ecg_indexdate: self.index_date})
            self.log.info(f"ECG_classresults - Renamed column {self.ecg_indexdate} to {self.index_date}")

        merged_ecg = df_detail.merge(
            df_classresults[
                [self.studyId, self.index_date, self.ecgtestId, self.acquisitiondate, self.incjclsinfarction]
            ],
            on=[self.studyId, self.ecgtestId],
            how="left",
        )

        row_count_before = merged_ecg.shape[0]
        merged_ecg = merged_ecg.drop_duplicates()
        row_count_after = merged_ecg.shape[0]
        self.log.info(f"Dropped {row_count_before - row_count_after} duplicate rows after merging")
        self.log.info(f"Remaining rows in merged ECG data: {row_count_after}")

        return merged_ecg

    def _st_deviation_index(self, df, region):
        """ST Deviation Index - captures overall ST segment abnormality
        Combines range (MaxST - MinST) with mean deviation (STM)
        """
        max_col = f"MaxST_{region}"
        min_col = f"MinST_{region}"
        mean_col = f"STM_{region}"

        # if any required column missing, set ST_deviation_index to NaN
        missing = [c for c in (max_col, min_col, mean_col) if c not in df.columns]
        if missing:
            self.log.debug(f"Setting ST_deviation_index for region {region} to NaN due to missing {missing}")
            df[f"ST_deviation_index_{region}"] = np.nan
            return

        df[f"ST_deviation_index_{region}"] = (np.abs(df[max_col] - df[min_col]) + np.abs(df[mean_col])) / 2

    def _compute_t_morphology_scores(self, df):
        """Compute T-wave morphology scores for all configured regions.

        Score per region = 0.4*|T_PeakAmpl_region| + 0.4*|T_Area_region| + 0.2*|T_Special_region|
        If none of the required columns exist for a region, the score is set to NaN.
        """
        try:
            for region in set(self.lead_regions.values()):
                amp_col = f"T_PeakAmpl_{region}"
                area_col = f"T_Area_{region}"
                special_col = f"T_Special_{region}"

                # If none of the columns exist, set score to NaN
                if not any(col in df.columns for col in (amp_col, area_col, special_col)):
                    df[f"T_morphology_score_{region}"] = np.nan
                    continue

                amp = df[amp_col].abs().fillna(0) if amp_col in df.columns else 0
                area = df[area_col].abs().fillna(0) if area_col in df.columns else 0
                special = df[special_col].abs().fillna(0) if special_col in df.columns else 0

                df[f"T_morphology_score_{region}"] = amp * 0.4 + area * 0.4 + special * 0.2

            return df

        except Exception as e:
            self.log.error(f"Error computing T-wave morphology scores: {e}")
            raise

    def _compute_regional_ischemia(self, df):
        """Compute a regional ischemia score for each configured region.

        Score per region = 0.4*|Q_PeakAmpl_region| + 0.4*ST_deviation_index_region + 0.2*T_morphology_score_region

        If none of the required columns exist for a region, the score is set to NaN.
        """
        try:
            for region in set(self.lead_regions.values()):
                q_col = f"Q_PeakAmpl_{region}"
                st_col = f"ST_deviation_index_{region}"
                t_col = f"T_morphology_score_{region}"
                out_col = f"Regional_ischemia_{region}"

                # If none of the columns exist, set score to NaN
                if not any(col in df.columns for col in (q_col, st_col, t_col)):
                    df[out_col] = np.nan
                    continue

                q = df[q_col].abs().fillna(0) if q_col in df.columns else 0
                st = df[st_col].fillna(0) if st_col in df.columns else 0
                t = df[t_col].fillna(0) if t_col in df.columns else 0

                df[out_col] = q * 0.4 + st * 0.4 + t * 0.2

            return df

        except Exception as e:
            self.log.error(f"Error computing regional ischemia scores: {e}")
            raise

    def create_composites(self, df_ecg):
        """Function creates composite ecg features per region"""
        df = df_ecg.copy()

        # compute st deviation index for all configured regions
        for region in set(self.lead_regions.values()):
            self._st_deviation_index(df, region)

        # compute t-wave morphology score for all regions
        for region in set(self.lead_regions.values()):
            self._compute_t_morphology_scores(df)

        # compute regional ischemia score
        for region in set(self.lead_regions.values()):
            self._compute_regional_ischemia(df)

        return df

    def process_ecg_data(self):
        """Executes the full ECG processing pipeline."""
        try:
            # Start timer for the whole pipeline
            start_time = time.time()
            self.log.info("Starting ECG processing pipeline.")

            # Step 1: Select ECG data
            step_start = time.time()
            df_ecg = self.validate_and_prepare_ecg_data()
            step_duration = time.time() - step_start
            self.log.info(f"ECG data selected with {df_ecg.shape[0]} rows.Time taken: {step_duration:.2f} seconds.")

            # Step 2: Process dynamic leads
            step_start = time.time()
            df_pivoted = self.transform_ecg_leads(df_ecg)
            step_duration = time.time() - step_start
            self.log.info(
                f"Dynamic ECG leads processed with {df_pivoted.shape[0]} rows.Time taken: {step_duration:.2f} seconds."
            )

            # Step 3: Replace zeros with NaN
            step_start = time.time()
            df_processed = self._process_zeros(df_pivoted)
            step_duration = time.time() - step_start
            self.log.info(
                f"ECG measurements processed with {df_processed.shape[0]} rows.Time taken: {step_duration:.2f} seconds."
            )

            # Step 4: Aggregate features by region
            step_start = time.time()
            df_region = self.group_ecg_by_region(df_processed, self.varying_vars)
            step_duration = time.time() - step_start
            self.log.info(
                f"ECG data grouped by region with {df_region.shape[0]} rows.Time taken: {step_duration:.2f} seconds."
            )

            # Total time for the whole pipeline
            total_duration = time.time() - start_time
            self.log.info(
                f"ECG processing pipeline completed successfully.Total time taken: {total_duration:.2f} seconds."
            )

            return df_region

        except Exception as e:
            self.log.error(f"Error during ECG processing pipeline: {str(e)}")
            raise
