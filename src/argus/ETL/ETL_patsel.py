import logging
import time

from argus.utils.config import load_config
from argus.utils.general_argus_functions import DataProcessingUtils


class PatientSelector:
    def __init__(self, dfPatsel, logger=None, version=None, mode=None):
        """Initialize with patient data and logging preference."""
        self.dfPatsel = dfPatsel.copy()
        self.version = version
        self.mode = mode

        # Set up logger
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

        # Load configuration
        self.config = load_config("general", version=version, mode=mode)
        self.studyId = self.config["columns"]["study_id"]
        self.age_col = self.config["columns"]["age"]
        self.gender_col = self.config["columns"]["gender"]
        self.index_date = self.config["columns"]["index_date"]  # = CT_Date

    def drop_missing_rows_patient_data(self, required_cols):
        """Validate processed patient data, remove invalid rows, and log issues."""
        try:
            # Drop rows with missing values in required columns
            missing_rows = self.dfPatsel[self.dfPatsel[required_cols].isnull().any(axis=1)]

            if not missing_rows.empty:
                missing_info = missing_rows[[self.studyId] + required_cols]
                self.log.debug(f"Removing {len(missing_rows)} rows due to missing values: {missing_info}")

            self.dfPatsel = self.dfPatsel.dropna(subset=required_cols)

        except Exception as e:
            self.log.error(f"Error removing missing rows in patient selection data: {str(e)}")
            raise

    def filter_adult_patients(self):
        """Filter patients to include only adults (age >= 18)."""
        try:
            self.dfPatsel[self.age_col] = self.dfPatsel[self.age_col].astype(int)
            invalid_age_rows = self.dfPatsel[~self.dfPatsel[self.age_col].between(18, 120)]
            invalid_count = len(invalid_age_rows)

            if invalid_count > 0:
                # Log details about removed rows
                invalid_info = invalid_age_rows[[self.studyId, self.age_col]].reset_index().to_dict(orient="records")
                self.log.info(f"Removing {invalid_count} rows due to Age out of range")
                self.log.debug(f"Removed rows: {invalid_info}")

                # Filter the dataframe in place
                self.dfPatsel = self.dfPatsel[self.dfPatsel[self.age_col].between(18, 120)]
            else:
                self.log.info("No rows removed due to age filtering.")

            return self.dfPatsel

        except Exception as e:
            self.log.error(f"Error filtering adult patients: {str(e)}")
            raise

    def process_patient_selection_data(self):
        """Process patient data: filter, map gender, convert dates, calculate age."""
        try:
            # Start timer for the whole pipeline
            start_time = time.time()
            self.log.info("Processing patient selection data started.")

            # Step 1: validate input type is pd.DataFrame
            required_cols = [
                self.studyId,
                self.gender_col,
                self.index_date,
                self.age_col,
            ]
            DataProcessingUtils.validate_input_data(self.dfPatsel, required_cols, id_col=self.studyId, logger=self.log)

            self.log.info(f"Input data validation successful. Required columns: {required_cols}")

            # Log the initial number of rows
            initial_count = len(self.dfPatsel)
            self.log.info(f"Initial number of rows in patient selection data: {initial_count}")

            # Step 2: Filter for adults (Age >= 18) and check Age <= 120
            self.dfPatsel = self.filter_adult_patients()

            # Step 3: Map gender
            self.log.info("Mapping gender column started.")

            self.dfPatsel = DataProcessingUtils.map_gender_values(self.dfPatsel, config=self.config, logger=self.log)

            # Step 4: Convert dates
            date_cols = [self.index_date]  # , 'deceasedDateTime', 'dob']
            self.dfPatsel = DataProcessingUtils.convert_date_columns(
                self.dfPatsel,
                date_cols,
                config=self.config,
                id_col=self.studyId,
                logger=self.log,
            )

            # Step 5: Drop missing rows in required columns
            self.drop_missing_rows_patient_data(required_cols)

            # Log the number of removed rows
            removed_count = initial_count - len(self.dfPatsel)
            self.log.info(f"Total rows removed: {removed_count}. Remaining rows:{len(self.dfPatsel)}.")
            nr_of_patients = self.dfPatsel[self.studyId].nunique()
            self.log.info(f"Number of unique patients remaining in the data: {nr_of_patients}")

            # Total time for the whole pipeline
            total_duration = time.time() - start_time
            self.log.info(
                f"Successfully Extracted, Transformed and Loaded patient selection"
                f"data. Time duration for this step: {total_duration:.2f} seconds."
            )

            return self.dfPatsel

        except Exception as e:
            self.log.error(f"Error processing patient selection data: {str(e)}")
            raise
