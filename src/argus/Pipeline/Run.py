"""Run the ARGUS inference pipeline and generate patient risk predictions."""

import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from joblib import load

from argus.Pipeline.Load_and_preprocess import LoadAndPreprocessor
from argus.utils.config import load_config
from argus.utils.general_functions import get_logs_dir, get_project_root
from argus.utils.logger import Logger


class RunInference:
    """Run Load and Data preprocessor and make predictions with ARGUS algorithm."""

    def __init__(self, version="v2_ARGUS25", mode="inference", logger=None):
        """Initialize the RunInference class.

        Args:
            version: Version identifier - default v2_ARGUS25.
            mode: "train" or "inference" - default inference.
            logger: Optional logger instance; defaults to the module logger.

        """
        self.version = version
        self.mode = mode

        # Create logger if not provided
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Load config for gender mappings
        self.config = load_config("general", version=self.version, mode=self.mode)
        self.gender_col = self.config["columns"]["gender"]
        self.gender_map = self.config["mappings"]["gender"]

        # Load models
        self.filename_model_female = self.config["models"]["model_f_filename"]
        self.filename_model_male = self.config["models"]["model_m_filename"]

    def load_and_preprocess(self, run_local=False, load_from_local=False):
        """Load source data, preprocess it, and return the merged pipeline output.

        Args:
            run_local (bool): Whether to run the preprocessing pipeline locally.
            load_from_local (bool): Whether to load preprocessed data from a local cache.

        Returns:
            dict: Dictionary containing the merged dataset and any other pipeline outputs.

        """
        preprocessor = LoadAndPreprocessor(
            version=self.version,
            mode=self.mode,
            run_local=run_local,
            load_from_local=load_from_local,
            logger=self.logger,
            from_date=self.from_date,
        )

        try:
            results = preprocessor.run_pipeline()
            self.logger.info(f"Processing completed. Final dataset shape: {results['df_merged_all'].shape}")
            return results

        except Exception as e:
            self.logger.error(f"Error in Load and Preprocess script with mode:{self.mode}. Error: {str(e)}")
            raise

    def predict_by_gender(self, df, gender_name) -> pd.DataFrame:
        """Predict risk scores for patients using the appropriate gender-specific model.

        Args:
            df (pd.DataFrame): DataFrame containing patient data
            gender_name: The gender to determine which model to use

        Returns:
            pd.DataFrame: DataFrame with added 'predicted_risk' and 'missing_fraction' columns

        """
        # Determine which model to use based on gender
        if gender_name == "female":
            model_filename = self.filename_model_female
        elif gender_name == "male":
            model_filename = self.filename_model_male
        else:
            raise ValueError(f"Unknown gender name: {gender_name} - must be 'female' or 'male'")

        # Construct path to model file
        # TODO change path for deployment
        model_filepath = Path(get_project_root()) / "models" / model_filename

        # Load the trained model
        self.logger.info(f"Loading {gender_name} model from {model_filepath}")

        model = load(model_filepath)
        self.logger.info(f"Loaded model {model_filename}")

        # Get feature columns expected by the model
        input_cols = list(model.feature_names_in_)
        input_cols = [str(c) for c in input_cols]

        # Prepare input features for prediction
        X = df[input_cols]

        self.logger.info("Start prediction")
        # Generate predictions (probability of class 1)
        prob_class_1 = model.predict_proba(X)[:, 1]

        # Add predictions and missing data fraction to dataframe
        df_result = df.copy()
        df_result["predicted_risk"] = prob_class_1
        df_result["missing_fraction"] = df_result.isnull().mean(axis=1)
        self.logger.info(f"Finished predictions for {model_filename}")

        return df_result

    def run(self, from_date: Optional[datetime.datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full inference pipeline for male and female patient cohorts.

        Args:
            from_date: Optional cutoff date used by the preprocessing pipeline.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Predicted risk data for female and male
                patients, respectively.

        """
        self.from_date = from_date

        # TODO - remove run_local and load to set to False
        results = self.load_and_preprocess(run_local=False, load_from_local=False)
        df = results["df_merged_all"]

        code_to_gender = {v: k for k, v in self.gender_map.items()}

        # Find codes for male and female
        female_code = None
        male_code = None

        for code, gender in code_to_gender.items():
            if gender.lower() in ["female"]:
                female_code = code
            elif gender.lower() in ["male"]:
                male_code = code

        if female_code is None or male_code is None:
            self.logger.error(f"Could not find gender code in config: {self.gender_map}")
            raise ValueError("Gender code not found in config")

        df_female = df[df[self.gender_col] == female_code].copy()
        df_male = df[df[self.gender_col] == male_code].copy()

        self.logger.info(f"Split data: {len(df_female)} female, {len(df_male)} male patients")

        # make predictions
        df_female = self.predict_by_gender(df_female, "female")
        df_male = self.predict_by_gender(df_male, "male")
        self.logger.info("finished predictions")

        return df_female, df_male


if __name__ == "__main__":
    # run script:
    logger = Logger(
        log_file=get_logs_dir() / "RunInference.log",
        enabled=True,
        df_level=logging.DEBUG,
    )

    try:
        processor = RunInference(mode="inference", logger=logger)

        df_female, df_male = processor.run()
        logger.info("Finished running")
        logger.debug(f"df_female looks like: {df_female.head(3)}")
        logger.debug(f"df_male looks like: {df_male.head(3)}")

    except Exception:
        logger.exception("RunInference crashed")
        raise

    finally:
        # ALWAYS persist dataframe logs
        df_log = logger.to_dataframe()
        # TODO: save df_log to Posit

    print("Finished run")
