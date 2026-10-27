"""Data loading and preprocessing pipeline for ARGUS.

This module orchestrates the patient-selection, laboratory, ECG, and
measurement preprocessing steps, then merges them into a single patient-level
DataFrame for downstream training or inference.
"""

import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from argus.ETL.merged_data import CompleteMerge, MergeECG, MergeLab, MergeMetingen
from argus.Pipeline.ETLPipeline_PatselProcessor import PatselProcessor
from argus.Pipeline.ETLPipeline_PreProcessor import (
    Body_and_BP_Processor,
    ECGProcessor,
    LabProcessor,
)
from argus.process.outlier_detection_metingen import MetingenOutlierDetector
from argus.utils.general_functions import get_logs_dir, get_project_root
from argus.utils.logger import Logger
from argus.utils.PathManager import PathManager_RES_Folder_UPOD


class LoadAndPreprocessor:
    """Complete data preprocessing pipeline for ARGUS.

    Handles loading, processing, and merging of patient selection,
    lab, ECG, and measurement data.

    1. Runs ETLPipeline_PatselProcessor to get patient selection
    2. Runs ETLPipeline_PreProcessor to get lab, ECG and measurement data
    3. Merges all dataset to get DataFrame with one row per patient

    """

    def __init__(
        self,
        version="v2_ARGUS25",
        mode="inference",
        run_local=False,
        load_from_local=False,
        logger=None,
        from_date: Optional[datetime.datetime] = None,
        example: Optional[bool] = None
    ):
        """Initialize the data preprocessor.

        Args:
            version: Version identifier - default v2_ARGUS25
            mode: "train" or "inference" - default inference
            run_local: Whether to save/load locally - use for development purposes
            load_from_local: Whether to load existing files - use for development purposes
            logger: logger instance (creates new one if None)
            from_date: Optional cutoff date used to filter data loaded into the pipeline.
            example: Optional boolean. Set to True when using example dataset.

        """
        self.version = version
        self.mode = mode
        self.run_local = run_local
        self.load_from_local = load_from_local
        self.from_date = from_date
        self.example = example

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.path_manager = PathManager_RES_Folder_UPOD(self.logger)

    def _get_local_path(self, filename, mode):
        return Path(get_project_root()) / "data" / "intermediates" / mode / filename

    def save_local(self, df, filename, mode=None):
        """Save dataframe to data/intermediates/{mode}/filename when run_local is True."""
        if mode is None:
            mode = self.mode
        if not self.run_local or df is None:
            return None
        try:
            output_path = self._get_local_path(filename, mode)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path)
            self.logger.info(f"Saved {filename} to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed saving {filename}: {e}")
            return None

    def load_local(self, filename, mode=None):
        """Load and return specified dataframe."""
        if mode is None:
            mode = self.mode
        if not self.run_local:
            return None
        if not self.load_from_local:
            self.logger.info(f"Skipping load for {filename} (load_from_local=False)")
            return None
        path = self._get_local_path(filename, mode)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            self.logger.info(f"Loaded {filename} from {path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed loading {filename}: {e}")
            return None

    def process_patsel(self):
        """Process patient selection data."""
        if self.load_from_local:
            df_patsel = self.load_local("dfPatsel_filtered.parquet")
        else:
            df_patsel = None

        if df_patsel is None:
            processor = PatselProcessor(
                logger=self.logger,
                mode=self.mode,
                version=self.version,
                save_intermediates=False,
                path_manager=self.path_manager,
                from_date=self.from_date,
                example = self.example
            )

            results = processor.main()
            df_patsel = results["dfPatsel_final"]
            stats = results.get("stats", {})
            self.logger.info(f"{stats}")

            self.save_local(df_patsel, "dfPatsel_filtered.parquet")

        return df_patsel

    def process_lab(self):
        """Process lab data."""
        if self.load_from_local:
            df_lab = self.load_local("dfLab.parquet")
        else:
            df_lab = None

        if df_lab is None:
            processor_lab = LabProcessor(
                logger=self.logger,
                mode=self.mode,
                version=self.version,
                path_manager=self.path_manager,
                from_date=self.from_date,
            )
            df_lab = processor_lab.main()
            self.save_local(df_lab, "dfLab.parquet")

        return df_lab

    def process_ecg(self):
        """Process ECG data."""
        if self.load_from_local:
            df_ecg = self.load_local("dfECG.parquet")
        else:
            df_ecg = None

        if df_ecg is None:
            processor_ecg = ECGProcessor(
                logger=self.logger,
                mode=self.mode,
                version=self.version,
                path_manager=self.path_manager,
                from_date=self.from_date,
            )
            df_ecg = processor_ecg.main()
            self.save_local(df_ecg, "dfECG.parquet")

        return df_ecg

    def process_measurements(self):
        """Process measurement data."""
        if self.load_from_local:
            df_processed_metingen = self.load_local("dfProcessedMetingen.parquet")
            df_processed_metingen_bp = self.load_local("dfProcessedMetingen_bp.parquet")
        else:
            df_processed_metingen = None
            df_processed_metingen_bp = None

        if df_processed_metingen is None or df_processed_metingen_bp is None:
            processor_measurements = Body_and_BP_Processor(
                logger=self.logger,
                mode=self.mode,
                version=self.version,
                path_manager=self.path_manager,
                from_date=self.from_date,
            )

            df_processed_metingen, df_processed_metingen_bp = processor_measurements.main()

            self.save_local(df_processed_metingen, "dfProcessedMetingen.parquet")
            self.save_local(df_processed_metingen_bp, "dfProcessedMetingen_bp.parquet")

        return df_processed_metingen, df_processed_metingen_bp

    def run_pipeline(self):
        """Run the complete preprocessing pipeline incl merging of datasets.

        Returns:
            dict: Dictionary containing all processed DataFrames.

        """
        self.logger.info(f"Starting preprocessing pipeline - mode: {self.mode}, version: {self.version}")

        # Process individual data sources
        df_patsel = self.process_patsel()
        df_lab = self.process_lab()
        df_ecg = self.process_ecg()
        df_processed_metingen, df_processed_metingen_bp = self.process_measurements()

        # Merge Lab data with Patsel
        merge_lab = MergeLab(df_patsel, df_lab, version=self.version, logger=self.logger, mode=self.mode)
        df_lab_merged = merge_lab.merge_lab_data()

        # Merge ECG data with Patsel
        ecg_merger = MergeECG(df_patsel, df_ecg, version=self.version, logger=self.logger, mode=self.mode)
        merged_ecg_data = ecg_merger.process_ecg_data()

        # Merge Metingen data with Patsel
        meting_merger = MergeMetingen(
            df_processed_metingen,
            df_patsel,
            df_processed_metingen_bp,
            version=self.version,
            logger=self.logger,
            mode=self.mode,
        )
        merged_meting_data = meting_merger.run()

        # Handle metingen outliers
        detector = MetingenOutlierDetector(
            config_file="outliers", version=self.version, logger=self.logger, mode=self.mode
        )
        metingen_processed = detector.run_pipeline(merged_meting_data)
        self.logger.info("Outlier handling for metingen completed successfully.")

        # Merge all data together
        merger = CompleteMerge(
            df_patsel,
            df_lab=df_lab_merged,
            df_ecg=merged_ecg_data,
            df_meting=metingen_processed,
            version=self.version,
            logger=self.logger,
            mode=self.mode,
        )

        merged_patsel_meting = merger.mergePatselMeting(df_patsel, metingen_processed)
        merged_patsel_meting_lab = merger.mergePatselMetingLab(merged_patsel_meting, df_lab_merged)
        df_merged_all = merger.mergePatselMetingLabECG(merged_patsel_meting_lab, merged_ecg_data)

        self.save_local(df_merged_all, f"dfPreprocessed_{self.mode}.parquet")

        self.logger.info(f"Pipeline completed. Final dataset shape: {df_merged_all.shape}")

        return {
            "df_patsel": df_patsel,
            "df_lab_merged": df_lab_merged,
            "df_ecg_merged": merged_ecg_data,
            "df_metingen_processed": metingen_processed,
            "df_merged_all": df_merged_all,
            "logger": self.logger,
        }

if __name__ == "__main__":
    # run direct in script
    run_full = False

    logger = Logger(log_file=get_logs_dir() / "run_preprocess.log", enabled=True)

    preprocessor = LoadAndPreprocessor(
        version="v2_ARGUS25",
        # mode="inference",
        mode="train",
        run_local=False,
        load_from_local=False,
        logger=logger,
        example=True
    )

    if run_full:
        results = preprocessor.run_pipeline()
        print(f"Processing completed. Final dataset shape: {results['df_merged_all'].shape}")
    else:
        # df_lab = preprocessor.process_lab() #use to proces lab data
        df_patsel = preprocessor.process_patsel() # test example dataset
