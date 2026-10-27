import logging
import os
import sys
import time

import pandas as pd

# Add src to path for imports
sys.path.append(os.path.abspath("../src"))

from argus.utils.config import load_config
from argus.utils.general_argus_functions import DataProcessingUtils


class MergeECG:
    def __init__(self, dfPatsel, dfECG, version=None, logger=None, mode=None):
        # Load configurations
        self.config_general = load_config("general", version=version, mode=mode)
        self.config = load_config("ecg", version=version, mode=mode)

        # Set up logger
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

        # Set parameters
        self.dfPatsel = dfPatsel
        self.dfECG = dfECG

        # Set column names and parameters
        self._set_column_names()
        self._set_ecg_parameters()

    def _set_column_names(self):
        self.studyId = self.config_general["columns"]["study_id"]
        self.gender = self.config_general["columns"]["gender"]
        self.index_date = self.config_general["columns"]["index_date"]
        self.age = self.config_general["columns"]["age"]

        self.acquisitiondate = self.config["columns_ecg"]["acquisition_date"]
        self.bitflag = self.config["columns_ecg"]["bitflag"]
        self.leadIdText = self.config["columns_ecg"]["leadIdText"]
        self.ecgtestId = self.config["columns_ecg"]["ecgtestId"]
        # self.ecg_indexdate = self.config["columns_ecg"]["index_date"]
        self.ecgdate = self.config["columns_ecg"]["ecgdate"]

    def _set_ecg_parameters(self):
        self.days_after = self.config["etl"]["ecg"]["days_after"]
        self.days_prior = self.config["etl"]["ecg"]["days_prior"]

    def flatten_ecg_data(self):
        """Flatten the ECG data by exploding non-index columns"""
        try:
            df = self.dfECG.copy()

            index_cols = [
                self.studyId,
                self.index_date,
                self.acquisitiondate,
                self.ecgtestId,
            ]

            # Reset index if it's a MultiIndex
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            # Get all columns except index columns
            cols_to_explode = [col for col in df.columns if col not in index_cols]

            # Explode the DataFrame
            flattened_df = df.explode(cols_to_explode)
            self.log.debug(f"Flatten dataframe looks like this {flattened_df.head()}")
            return flattened_df

        except Exception as e:
            self.log.error(f"Error flattening ECG data: {str(e)}")
            raise

    def merge_ecg_date(self):
        """Merge ECG data with patient selection data"""
        try:
            # Check if the necessary columns are in dfPatsel
            if self.studyId not in self.dfPatsel.columns or self.index_date not in self.dfPatsel.columns:
                self.log.error((f"Missing necessary columns in dfPatsel: {self.studyId} or {self.index_date}"))
                raise ValueError(f"Missing necessary columns in dfPatsel: {self.studyId} or {self.index_date}")

            # Flatten the ECG data
            flat_dfECG = self.flatten_ecg_data()
            self.log.info(f"Flattened ECG data shape: {flat_dfECG.shape}")

            flat_dfECG = flat_dfECG.loc[:, ~flat_dfECG.columns.duplicated()]

            # Make sure datetime columns are set
            flat_dfECG[f"{self.index_date}"] = pd.to_datetime(flat_dfECG[self.index_date]).dt.strftime("%Y-%m-%d")

            self.dfPatsel[self.index_date] = pd.to_datetime(self.dfPatsel[self.index_date]).dt.strftime("%Y-%m-%d")

            # alreadye done
            # flat_dfECG = flat_dfECG.rename(
            #     columns={f'{self.ecg_indexdate}_date': self.index_date}
            # )

            # Merge ECG data with patient data (dfPatsel)
            merged_ecg = pd.merge(
                self.dfPatsel[[self.studyId, self.index_date]],
                flat_dfECG,
                on=[self.studyId, self.index_date],
            )

            # Log the shape after merging
            self.log.info(f"Rows after merging with dfPatsel: {merged_ecg.shape[0]}")
            return merged_ecg

        except Exception as e:
            self.log.error(f"Error during merging ecg data with patsel: {str(e)}")
            raise

    def clean_merged_ecg_data(self, merged_ecg):
        """Clean and process the merged ECG data"""
        try:
            cleaned_ecg = merged_ecg.copy()

            # Split the acquisition date into date and time components
            cleaned_ecg = DataProcessingUtils.create_time_features(cleaned_ecg, self.acquisitiondate, logger=self.log)
            ecg_date_col = f"{self.acquisitiondate}_date"
            cleaned_ecg[self.ecgdate] = cleaned_ecg[ecg_date_col]

            # Convert dates to datetime for calculation
            cleaned_ecg[self.index_date] = pd.to_datetime(cleaned_ecg[self.index_date])
            cleaned_ecg[self.ecgdate] = pd.to_datetime(cleaned_ecg[self.ecgdate])

            # Calculate days until CT scan
            cleaned_ecg["meta_days_until_ct"] = (cleaned_ecg[self.index_date] - cleaned_ecg[self.ecgdate]).dt.days

            # Filter by time window: update 2026-05-06
            cleaned_ecg = cleaned_ecg[
                (cleaned_ecg["meta_days_until_ct"] < self.days_prior)
                & (cleaned_ecg["meta_days_until_ct"] > -self.days_after)
            ]

            # cleaned_ecg = cleaned_ecg[
            #     cleaned_ecg["meta_days_until_ct"].between(
            #         -self.days_after, self.days_prior
            #     )
            # ]

            # Log the shape after filtering
            self.log.info(f"Rows after filtering by time window: {cleaned_ecg.shape[0]}")

            # Keep only the ECG measurement closest to the CT scan
            idx = cleaned_ecg.groupby([self.studyId, self.index_date])["meta_days_until_ct"].apply(
                lambda x: x.abs().idxmin()
            )

            cleaned_ecg = cleaned_ecg.loc[idx]
            self.log.info(f"Final rows after keeping closest measurements: {cleaned_ecg.shape[0]}")
            # Drop unnecessary columns
            cols_to_drop = [
                f"{self.acquisitiondate}_time",
                f"{self.acquisitiondate}_date",
                "meta_days_until_ct",
                self.ecgdate,
            ]

            cleaned_ecg = cleaned_ecg.drop(columns=cols_to_drop, errors="ignore")
            cleaned_ecg[self.index_date] = pd.to_datetime(cleaned_ecg[self.index_date], errors="coerce")

            return cleaned_ecg

        except Exception as e:
            self.log.error(f"Error during cleaning merged ECG data: {str(e)}")
            raise

    def process_ecg_data(self):
        """Start process of merging ECG data with patient selection"""
        try:
            self.log.info("Starting process of merging ECG data with patient selection")

            # Merge ECG data with patient data
            merged_data = self.merge_ecg_date()
            self.log.info("Completed merging ECG data with patient data")

            # Clean and process the merged data
            processed_data = self.clean_merged_ecg_data(merged_data)
            processed_data = processed_data.drop_duplicates()
            self.log.info(
                f"After processing and dropping duplicates we have {processed_data[self.studyId].nunique()} patients with ECG data"
            )
            self.log.info("Completed cleaning and processing merged ecg data")

            return processed_data

        except Exception as e:
            self.log.error(f"Error in ecg-patsel processing workflow: {str(e)}")
            raise


class MergeLab:
    def __init__(self, dfPatsel, combined_lab_df, version=None, logger=None, mode=None):
        """Initializes the MergeLab class with patient and lab DataFrames.

        Args:
            dfPatsel (pd.DataFrame): DataFrame containing patient selection data.
            combined_lab_df (pd.DataFrame): Combined lab data with all features.
            version (str, optional): Version identifier for configuration.
            logging_enabled (bool, optional): Whether to enable logging.

        """
        self.dfPatsel = dfPatsel.copy()
        self.combined_lab_df = combined_lab_df.copy()
        self.version = version

        # Load configurations
        self.config = load_config("lab")
        self.config_general = load_config("general", version=version, mode=mode)

        # Set general column names
        self.studyId = self.config_general["columns"]["study_id"]
        self.index_date = self.config_general["columns"]["index_date"]
        self.gender = self.config_general["columns"]["gender"]
        self.age = self.config_general["columns"]["age"]

        # Set lab specific column names
        self.lab_test_date = self.config["etl"]["columns"]["lab_test_date"]

        # Set time window parameters
        self.days_after = self.config["etl"]["lab"]["days_after"]
        self.days_prior = self.config["etl"]["lab"]["days_prior"]

        # Set up logger
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

    def prepare_dataframes(self):
        """Prepares the DataFrames for merging by ensuring datetime types and creating
        necessary columns.

        Returns:
            tuple: Prepared patient DataFrame and lab DataFrame.

        """
        self.log.info("Preparing DataFrames for merging.")

        # Prepare patient DataFrame
        patsel_df = self.dfPatsel.copy()
        patsel_df[self.index_date] = pd.to_datetime(patsel_df[self.index_date])

        # Ensure required columns exist
        required_cols = [self.studyId, self.index_date, self.age, self.gender]
        DataProcessingUtils.validate_input_data(patsel_df, required_cols, id_col=self.studyId, logger=self.log)

        # Prepare lab DataFrame
        lab_df = self.combined_lab_df.copy()
        lab_df[self.lab_test_date] = pd.to_datetime(lab_df[self.lab_test_date])

        # Ensure lab DataFrame has studyId and test date
        DataProcessingUtils.validate_input_data(
            lab_df,
            [self.studyId, self.lab_test_date],
            id_col=self.studyId,
            logger=self.log,
        )

        self.log.info("DataFrames prepared successfully.")
        return patsel_df, lab_df

    def filter_by_date_range(self, patsel_df, lab_df):
        """Filters lab data to only include tests within the specified time range around
        the index/ consultation/ CT scan date.
        """
        self.log.info(
            f"Filtering lab data to within {self.days_prior} days beforeand {self.days_after} days after index date."
        )

        # Merge labs with patient data on studyId
        merged_df = pd.merge(
            patsel_df[[self.studyId, self.index_date, self.age, self.gender]],
            lab_df,
            on=self.studyId,
            how="inner",
        )

        # Calculate days between lab test and CT scan
        merged_df["days_from_ct"] = (merged_df[self.index_date] - merged_df[self.lab_test_date]).dt.days

        # Filter to only include tests within the time window
        filtered_df = merged_df[
            (merged_df["days_from_ct"] < self.days_prior) & (merged_df["days_from_ct"] > -self.days_after)
        ]

        before_count = len(merged_df)
        after_count = len(filtered_df)
        removed_count = before_count - after_count

        self.log.info(f"Filtered lab data: {removed_count} rows removed{after_count} rows remain within time window.")

        return filtered_df

    def select_closest_lab_values(self, filtered_df):
        """Selects the lab test closest to the index date for each patient-date
        combination.Each patient can have multiple lab tests, so we need to
        find the one closest to the index date.
        """
        self.log.info("Selecting lab values closest to index date.")

        # Identify lab feature columns (all columns except the metadata ones)
        metadata_cols = [
            self.studyId,
            self.index_date,
            self.lab_test_date,
            self.age,
            self.gender,
            "days_from_ct",
        ]
        feature_cols = [col for col in filtered_df.columns if col not in metadata_cols]

        # Take absolute value of days_from_ct to find closest lab test
        filtered_df["abs_days_from_ct"] = filtered_df["days_from_ct"].abs()

        # For each patient and index_date, find the row with the minimum date
        # difference (i.e., the closest lab test)
        closest_indices = filtered_df.groupby([self.studyId, self.index_date])["abs_days_from_ct"].idxmin()

        # Select those rows from the filtered DataFrame
        closest_labs = filtered_df.loc[closest_indices]

        # Select only the needed columns
        result_df = closest_labs[
            [
                self.studyId,
                self.index_date,
                self.age,
                self.gender,
                self.lab_test_date,
                # "days_from_ct",
            ]
            + feature_cols
        ].copy()

        self.log.info(f"Selected closest lab values. Result has {len(result_df)} rows.")

        return result_df

    def merge_lab_data(self):
        """Performs the full merge process from preparation to weighted averaging.

        Returns:
            pd.DataFrame: DataFrame with patient data and weighted average lab values.

        """
        try:
            start_time = time.time()
            self.log.info("Starting lab data merging process.")

            # Step 1: Prepare DataFrames
            step_start = time.time()
            patsel_df, lab_df = self.prepare_dataframes()
            step_duration = time.time() - step_start
            self.log.info(f"DataFrames prepared. Time taken: {step_duration:.2f} seconds.")

            # Step 2: Filter by date range
            step_start = time.time()
            filtered_df = self.filter_by_date_range(patsel_df, lab_df)
            step_duration = time.time() - step_start
            self.log.info(
                f"Date range filtering complete. Time taken: {step_duration:.2f} seconds."  # noqa: E501
            )

            # Step 3: Apply weighted averaging
            step_start = time.time()
            merged_df = self.select_closest_lab_values(filtered_df)
            step_duration = time.time() - step_start
            self.log.info(
                f"Selecting closest lab values complete. Time taken: {step_duration:.2f} seconds."  # noqa: E501
            )

            # Step 4: Add patients without lab data back to the results
            final_df = pd.merge(
                patsel_df[[self.studyId, self.index_date, self.age, self.gender]],
                merged_df,
                on=[self.studyId, self.index_date, self.age, self.gender],
                how="left",
            )

            # Round lab values to 2 decimal place
            lab_cols = [
                col for col in final_df.columns if col not in [self.studyId, self.index_date, self.age, self.gender]
            ]
            final_df[lab_cols] = final_df[lab_cols].round(2)

            # convert to date
            final_df[self.index_date] = pd.to_datetime(final_df[self.index_date], errors="coerce").dt.date

            # Update the dataframe to drop the rows where all lab values are NaN
            final_df = final_df.dropna(subset=lab_cols, how="all")
            # Drop duplicates
            final_df = final_df.drop_duplicates()

            # Log completion metrics
            patients_with_labs = final_df.dropna(subset=lab_cols, how="all")[self.studyId].nunique()

            total_patients = final_df[self.studyId].nunique()
            self.log.info(f"Merging complete.{patients_with_labs} out of {total_patients} patients have lab data.")

            total_duration = time.time() - start_time
            self.log.info(f"Lab data merging process completed.Total time taken: {total_duration:.2f} seconds.")

            return final_df

        except Exception as e:
            self.log.error(f"Error in merging lab data: {str(e)}")
            raise


class MergeMetingen:
    """Class for merging processed metingen data with patient selection data."""

    def __init__(self, df_metingen, df_patsel, df_meting_bp=None, version=None, logger=None, mode=None):
        # Set up logger
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

        self.log.info("Initializing MergeMetingen class.")

        # Set parameters
        self.df_metingen = df_metingen.copy()
        self.df_patsel = df_patsel.copy()

        if df_meting_bp is not None:
            self.df_meting_bp = df_meting_bp.copy()
        else:
            self.log.info("Blood Pressure data is NOT available")
            self.df_meting_bp = None

        self.version = version
        self.df_meting_combined = None

        # Load configurations
        self.config = load_config("metingen", version=version, mode=mode)
        self.config_general = load_config("general", version=version, mode=mode)

        # Set general column names
        self.studyId = self.config_general["columns"]["study_id"]
        self.gender = self.config_general["columns"]["gender"]
        self.index_date = self.config_general["columns"]["index_date"]
        self.age = self.config_general["columns"]["age"]

        # Set metingen column names
        self.test_date = self.config["columns"]["meting_test_date"]
        self.max_days = self.config["metingen"]["max_days"]
        self.min_days = self.config["metingen"]["min_days"]  # added on 20-04-2026

    def merge_initial(self, df):
        """Perform merge to combine index date with metingen info."""
        self.log.info("Performing initial merge to add index date from patsel to metingen data.")
        try:
            # First merge to get the index_date into the metingen dataframe
            merged_df = pd.merge(
                df,
                self.df_patsel[[self.studyId, self.index_date]],
                on=self.studyId,
                how="inner",
            )

            self.log.info(f"Initial merge completed:{merged_df[self.studyId].nunique()} unique patients.")
            return merged_df

        except Exception as e:
            self.log.error(f"Error in initial merge: {e}")
            raise

    def calculate_time_difference(self, df):
        """Calculate the time difference between index_date and test_date."""
        self.log.info("Calculating time difference between index date and test date.")
        try:
            df[self.index_date] = pd.to_datetime(df[self.index_date], errors="coerce")
            df[self.test_date] = pd.to_datetime(df[self.test_date], errors="coerce")
            # remove abs on 2026-04-08: df["time_diff"] = abs(df[self.index_date] - df[self.test_date]).dt.days
            df["time_diff"] = (df[self.index_date] - df[self.test_date]).dt.days
            self.log.info("Time difference calculation complete.")
            return df
        except Exception as e:
            self.log.error(f"Error calculating time difference: {e}")
            raise

    def select_closest_records(self, df):
        """Find the closest record for each subject and index_date combination."""
        self.log.info("Selecting closest records to the index_date.")
        try:
            # Find the closest meting record for each subject-indexdate combination
            df_closest = df.loc[df.groupby([self.studyId, self.index_date])["time_diff"].idxmin()]
            # Debug log the selected records
            debug_cols = [self.studyId, self.index_date, self.test_date, "time_diff"]
            self.log.debug(f"Selected closest records for {self.studyId}: {df_closest[debug_cols].to_string()}")
            self.log.info(f"Closest records selected: {df_closest[self.studyId].nunique()} unique patients.")

            return df_closest
        except Exception as e:
            self.log.error(f"Error selecting closest records: {e}")
            raise

    def filter_by_time_window(self, df):
        """Drop entries older than specified days."""
        self.log.info(f"Filtering entries by time window (max {self.max_days} days).")
        try:
            # Filter based on the time_diff column
            # filtered_df = df[df["time_diff"] < self.max_days] # changed on 2026_04_08
            # updated on 2026_05_05, added the - sign before self.min_days:
            filtered_df = df[(df["time_diff"] >= -self.min_days) & (df["time_diff"] < self.max_days)]

            # Log details about discarded rows
            # updated on 2026_05-05 the discarded log. Before it just printed the length of the selected df
            discarded_df = df[(df["time_diff"] < -self.min_days) & (df["time_diff"] >= self.max_days)]

            if not discarded_df.empty:
                self.log.info(
                    f"Discarded {discarded_df.shape[0]} rows and "
                    f"{discarded_df[self.studyId].nunique()} patients "
                    f"with time difference > {self.max_days} days. "
                    f"Time differences ranged from {discarded_df['time_diff'].min()} "
                    f"to {discarded_df['time_diff'].max()} days."
                )
            else:
                self.log.info("No rows were discarded based on time window criteria.")
            self.log.info(f"Filtered by time window: {filtered_df[self.studyId].nunique()} patients remaining.")
            return filtered_df
        except Exception as e:
            self.log.error(f"Error filtering by time window: {e}")
            raise

    def merge_final_data(self, processed_df):
        """Merges the processed metingen data with complete patient selection data."""
        self.log.info("Performing merge with patient selection data.")
        try:
            # Determine which additional columns to get from patient selection
            try:
                age_col = self.config_general["columns"]["age"]
            except KeyError:
                age_col = "Age"

            additional_cols = [self.gender, age_col]
            available_cols = [col for col in additional_cols if col in self.df_patsel.columns]

            # Include studyId and index_date plus available additional columns
            patsel_cols = [self.studyId, self.index_date] + available_cols

            # Make sure datetime columns are set
            processed_df[f"{self.index_date}"] = pd.to_datetime(processed_df[self.index_date]).dt.strftime("%Y-%m-%d")

            self.df_patsel[self.index_date] = pd.to_datetime(self.df_patsel[self.index_date]).dt.strftime("%Y-%m-%d")

            # Final merge to bring in all patient data
            self.df_meting_combined = pd.merge(
                processed_df,
                self.df_patsel[patsel_cols],
                on=[self.studyId, self.index_date],
                how="inner",
            )

            self.log.info(f"Patsel merge completed: {self.df_meting_combined.shape[0]} rows.")
            return self.df_meting_combined

        except Exception as e:
            self.log.error(f"Error in patsel merge: {e}")
            raise

    def process_measurements(self, df_input, measurement_type="regular"):
        """Process measurement data pipeline.

        Returns:
            DataFrame: Processed measurement data ready for final merge

        """
        try:
            self.log.info(f"Starting measurement processing pipeline for {measurement_type} measurements.")

            # Step 1: Initial merge to get index_date into metingen data
            step_name = f"Initial merge for {measurement_type} measurements"
            merged_initial = self.merge_initial(df_input)

            # Step 2: Calculate time difference
            step_name = f"Calculate time difference for {measurement_type} measurements"
            with_time_diff = self.calculate_time_difference(merged_initial)

            # steps turned around on 2026-04-08
            # # Step 3: Select closest records
            # step_name = f"Select closest records for {measurement_type} measurements"
            # closest_records = self.select_closest_records(with_time_diff)

            # # Step 4: Filter by time window
            # step_name = f"Filter by time window for {measurement_type} measurements"
            # filtered_data = self.filter_by_time_window(closest_records))

            # Step 3: Filter by time window
            step_name = f"Filter by time window for {measurement_type} measurements"
            filtered_data = self.filter_by_time_window(with_time_diff)

            # Step 4: Select closest records
            step_name = f"Select closest records for {measurement_type} measurements"
            closest_records = self.select_closest_records(filtered_data)

            # Step 5: Final merge to get all patient data
            step_name = f"Final merge for {measurement_type} measurements"
            final_df = self.merge_final_data(closest_records)

            self.log.info(
                f"Processing pipeline completed for {measurement_type} measurements: {final_df[self.studyId].nunique()} patients."
            )
            return final_df

        except Exception as e:
            self.log.error(f"Error during {measurement_type} measurement processing at step '{step_name}': {e}")
            raise

    def merge_measurement_results(self, df_regular, df_bp=None):
        """Merge the results from regular and BP measurement processing.

        Returns:
            DataFrame: Combined measurement data

        """
        if df_bp is None:
            self.log.info("Only regular measurements available, returning regular measurements.")
            return df_regular

        try:
            self.log.info("Merging regular and blood pressure measurement results.")

            # Merge regular and BP measurements
            combined_df = pd.merge(
                df_regular,
                df_bp,
                on=[self.studyId, self.index_date, self.gender],
                how="outer",
                suffixes=("_regular", "_bp"),
            )

            # changed on 2025-05-05
            combined_df[self.age] = combined_df[f"{self.age}_regular"].combine_first(combined_df[f"{self.age}_bp"])
            # combined_df = combined_df.rename(columns={f"{self.age}_regular": self.age})

            self.log.info(f"Measurement results merged: {combined_df.shape[0]} rows.")
            return combined_df

        except Exception as e:
            self.log.error(f"Error merging measurement results: {e}")
            raise

    def get_final_dataframe(self):
        """Returns the final combined dataframe."""
        if self.df_meting_combined is not None:
            return self.df_meting_combined
        else:
            self.log.error("The data has not been processed and merged yet.")
            raise ValueError("The final DataFrame is not available because the merge pipeline has not been executed.")

    def run(self):
        """Runs the entire merge processing pipeline."""
        try:
            self.log.info("Starting the merge processing pipeline.")

            # Process regular measurements
            processed_regular = None
            if self.df_metingen is not None:
                self.log.info("Regular measurement data available, processing it.")
                processed_regular = self.process_measurements(self.df_metingen, "regular")

            # Process blood pressure measurements if available
            processed_bp = None
            if self.df_meting_bp is not None:
                self.log.info("Blood pressure data available, processing it.")
                processed_bp = self.process_measurements(self.df_meting_bp, "blood_pressure")

            # Merge the results
            self.df_meting_combined = self.merge_measurement_results(processed_regular, processed_bp)

            self.log.info("Merge pipeline completed successfully.")
            self.log.info(f"Nr. of rows in regular measurements: {processed_regular.shape[0]}")
            self.log.info(f"Nr. of patients in regular measurements: {processed_regular[self.studyId].nunique()}")
            self.log.info(f"Nr. of rows in bp measurements: {processed_bp.shape[0]}")
            self.log.info(f"Nr. of patients in bp measurements: {processed_bp[self.studyId].nunique()}")
            return self.get_final_dataframe()

        except Exception as e:
            self.log.error(f"Error during the merge pipeline execution: {e}")
            raise


class CompleteMerge:
    def __init__(self, df_patsel, df_lab=None, df_ecg=None, df_meting=None, version=None, logger=None, mode=None):
        # Set up logger
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

        self.log.info("Initializing MergeMetingen class.")

        self.version = version
        self.mode = mode

        # Load configurations
        self.config_general = load_config("general", version=version, mode=mode)
        self.config_meting = load_config("metingen", version=version, mode=mode)

        # Set general column names
        self.studyId = self.config_general["columns"]["study_id"]
        self.gender = self.config_general["columns"]["gender"]
        self.index_date = self.config_general["columns"]["index_date"]
        self.age = self.config_general["columns"]["age"]

        self.bmi = self.config_meting["columns"]["meting_bmi"]
        self.bp_dia = self.config_meting["columns"]["meting_bp_dia"]
        self.bp_syst = self.config_meting["columns"]["meting_bp_syst"]

        self.log.info("Number of patients in the datasets:")
        self.log.info(f"Patsel nr: {df_patsel[self.studyId].nunique()}")
        self.log.info(
            f"Lab nr: {df_lab[self.studyId].nunique()} and all Lab data in Patsel: {df_lab[self.studyId].isin(df_patsel[self.studyId]).all()}"
        )
        self.log.info(
            f"ECG nr: {df_ecg[self.studyId].nunique()} and all ECG data in Patsel: {df_ecg[self.studyId].isin(df_patsel[self.studyId]).all()}"
        )
        self.log.info(
            f"meting nr: {df_meting[self.studyId].nunique()} and all metingen data in Patsel: {df_meting[self.studyId].isin(df_patsel[self.studyId]).all()}"
        )

    def mergePatselMeting(self, df_patsel, df_meting):
        try:
            # Make sure Age is an int
            df_meting[self.age] = df_meting[self.age].astype("Int64")

            df = df_patsel[[self.studyId, self.gender, self.index_date, self.age]].merge(
                df_meting[[self.studyId, self.index_date, self.gender, self.age, self.bmi, self.bp_dia, self.bp_syst]],
                on=[self.studyId, self.index_date, self.gender, self.age],
                how="left",
            )

            # Convert to datetime
            # df[self.index_date] = pd.to_datetime(df[self.index_date], errors="coerce")

            self.log.info("Successfully merged Patsel with Metingen")
            return df

        except Exception as e:
            self.log.error(f"Error in Complete merge [patsel with meting]: {e}")
            return None

    def mergePatselMetingLab(self, df, df_lab):
        try:
            df[self.index_date] = pd.to_datetime(df[self.index_date], errors="coerce").dt.date
            df_lab[self.index_date] = pd.to_datetime(df_lab[self.index_date], errors="coerce").dt.date

            self.log.info(f"We have {df_lab[self.studyId].nunique()} number of patients with lab data")
            self.log.info(f"Being added to {df[self.studyId].nunique()} number of patients in the patient selection")
            self.log.info(
                f"So we have {df[self.studyId].nunique() - df_lab[self.studyId].nunique()} patients without any lab data"
            )

            df_new = df.merge(df_lab, on=[self.studyId, self.index_date, self.gender, self.age], how="left")

            self.log.info("Successfully merged Lab with Patsel + Metingen")
            return df_new

        except Exception as e:
            self.log.error(f"Error in Complete merge [Lab with patsel + meting]: {e}")
            return None

    def mergePatselMetingLabECG(self, df, df_ecg):
        try:
            # Convert to datetime
            df_ecg[self.index_date] = pd.to_datetime(df_ecg[self.index_date], errors="coerce")
            df[self.index_date] = pd.to_datetime(df[self.index_date], errors="coerce")

            expected_cols = [
                self.studyId,
                self.index_date,
                "InjClsInfarction",
                "ST_deviation_index_Anterior",
                "ST_deviation_index_Inferior",
                "ST_deviation_index_Lateral",
                "T_morphology_score_Anterior",
                "T_morphology_score_Inferior",
                "T_morphology_score_Lateral",
                "Regional_ischemia_Anterior",
                "Regional_ischemia_Inferior",
                "Regional_ischemia_Lateral",
            ]

            df_new = df.merge(df_ecg[expected_cols], on=[self.studyId, self.index_date], how="left")

            # make sure is numeric
            df_new["InjClsInfarction"] = pd.to_numeric(df_new["InjClsInfarction"], errors="coerce")

            self.log.info("Successfully merged ECG with Patsel + Metingen + Lab")
            return df_new

        except Exception as e:
            self.log.error(f"Error in Complete merge [Lab with patsel + meting]: {e}")
            return None
