import datetime
import functools
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from spacy.lang.nl import stop_words

from argus.data_loaders.factory import create_decursus_loader, create_patsel_loader
from argus.ETL import ETL_pob_filter, ETL_vg_filter
from argus.utils.config import load_config
from argus.utils.general_functions import get_project_root
from argus.utils.logger import Logger
from argus.utils.PathManager import PathManager_RES_Folder_UPOD

STOPWORDS = stop_words.STOP_WORDS


def timer_log_method(logger_attr: str = "logger"):
    """Decorator factory that logs duration to an instance logger if present,
    otherwise falls back to an appropriate module/class logger.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            # Try to find a logger on the first positional argument (usually `self`)
            obj = args[0] if args else None
            logger = None
            if obj is not None and hasattr(obj, logger_attr):
                logger = getattr(obj, logger_attr)
            if logger is None:
                # Fallback: prefer the object's module, else the function module
                name = obj.__class__.__module__ if obj is not None else func.__module__
                logger = logging.getLogger(name)
            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                # Use logger.info, but guard if logger missing method
                if hasattr(logger, "info"):
                    logger.info(f"{func.__name__} took {duration:.4f} seconds")
                else:
                    # Last-resort fallback to root logger
                    logging.getLogger().info(f"{func.__name__} took {duration:.4f} seconds")

        return wrapper

    return decorator


class PatselProcessor:
    """PatselProcessor handles the complete ETL workflow for patient selection data.

    This class processes the first steps of the ETL pipeline:
    1. Loading and processing patsel data using PatientSelector
    2. Loading decursus (consult) data
    3. Applying POB and VG filters to identify relevant patients
        --> (POB positive AND VG negative)
    4. Merging filtered data with patsel data
    5. Saving intermediate and final results (with resume capability?)

    Returns:
            pd.DataFrame: Final filtered patient selection data merged with
                decursus data.

    """

    def __init__(
        self,
        logger: Logger,
        mode: str,  # 'train', 'inference'
        version: str = "v2_ARGUS25",
        save_intermediates: Optional[bool] = None,
        debug_mode: Optional[bool] = None,
        from_date: Optional[datetime.datetime] = None,
        example: Optional[bool] = None,
        # Training params
        path_manager: Optional[PathManager_RES_Folder_UPOD] = None,
        # Inference params
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
    ) -> None:
        """Initialize PatselProcessor with configuration loading and validation."""
        # Set up logger
        self.logger = logger
        self.save_intermediates = save_intermediates
        self.mode = mode  # 'train', 'inference'
        self.version = version
        self.debug_mode = debug_mode
        self.example = example

        # Create data loader using factory
        self.patsel_loader = create_patsel_loader(
            mode=self.mode, logger=self.logger, version=self.version, path_manager=path_manager, from_date=from_date
        )

        self.decursus_loader = create_decursus_loader(
            mode=self.mode,
            logger=self.logger,
            version=self.version,
            path_manager=path_manager,
            # db_user=db_user,
            # db_password=db_password,
            debug_mode=self.debug_mode,
            from_date=from_date,
        )

        # Load configuration
        self.config = load_config("general", version=version, mode=self.mode)
        self.studyId = self.config["columns"]["study_id"]
        self.index_date = self.config["columns"]["index_date"]

        self.path_manager = path_manager if path_manager is not None else PathManager_RES_Folder_UPOD(self.logger)

        # load decursus and filter specific configurations:
        self.filter_config = load_config("patsel_processor")
        self.consult_id = self.filter_config["columns"]["consult_id_col"]
        self.index_date_input_col = self.filter_config["columns"]["index_date_input_col"]
        self.section_text = self.filter_config["columns"]["section_text_col"]
        self.section_title = self.filter_config["columns"]["section_title_col"]
        self.pob_indication = self.filter_config["columns"]["pob_indication_col"]
        self.vg_indication = self.filter_config["columns"]["vg_indication_col"]

    @timer_log_method(logger_attr="logger")
    def load_patsel_data(self) -> pd.DataFrame:
        """Load patsel data using the appropriate loader"""
        return self.patsel_loader.load_data()  # data_loader.base.py

    @timer_log_method(logger_attr="logger")
    def load_decursus_data(self) -> pd.DataFrame:
        """Load decursus data using the appropriate loader"""
        return self.decursus_loader.load_data()

    def ensure_string_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            return df
        if pd.api.types.is_string_dtype(df[col]):
            return df
        df = df.copy()
        try:
            df[col] = df[col].astype("int64").astype("str")
        except Exception:
            df[col] = df[col].astype("str")
        return df

    @timer_log_method(logger_attr="logger")
    def filter_pob(self, df_decursus):
        try:
            # prepare clinlp for selecting patients with "pijn op de borst" or
            # "kortademigheid"
            pob_filter = ETL_pob_filter.ETLPobFilter(
                config_path="pob_filter",
                use_meta_qualifiers=True,
                use_transformers=True,
            )
            pob_filter.prepare_clinlp()

            pob_filter.iterative = True
            pob_filter.n_jobs = 1  # Because we work in windows..
            pob_filter.batch_size = 512

            df_decursus[self.pob_indication] = pob_filter.get_POB_bulk(df_decursus[self.section_text])

            self.logger.debug(f"First POB indication: {df_decursus[self.pob_indication].iloc[0]}")

            df_decursus.section_text = df_decursus.section_text.str.replace(r"[\r\n]+", " ", regex=True)
            df_decursus.section_text = df_decursus.section_text.str.replace(r"\s{2,}", " ", regex=True)
            self.logger.info("Decursus data succesfully filtered for pob symptoms")

            if self.save_intermediates:
                folder = f"intermediates/{self.mode}"
                output_path = Path(get_project_root()) / "data" / folder / "patsel_pob.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_decursus.to_parquet(output_path)
                self.logger.info(f"Saved df_decursus pob to {output_path}")

            return df_decursus

        except Exception as e:
            self.logger.error(f"Error in pob filter on decursus data: {e}")
            return None

    @timer_log_method(logger_attr="logger")
    def vg_filter(self, df_decursus):
        try:
            self.logger.info("Starting VG filtering")
            vg_filter = ETL_vg_filter.ETLVGFilter(config_path="vg_filter", use_meta_qualifiers=True)
            vg_filter.use_meta_qualifiers = True
            vg_filter.prepare_clinlp()

            df_decursus[self.vg_indication] = df_decursus[self.section_text].apply(vg_filter.get_VG)

            self.logger.info("Decursus data succesfully filtered for voorgeschiedenis (vg)")

            self.logger.debug(f"Decursus dataframe now looks like this: {df_decursus.head(3)}")

            if self.save_intermediates:
                folder = f"intermediates/{self.mode}"
                output_path = Path(get_project_root()) / "data" / folder / "patsel_pob_vg_indicators.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_decursus.to_parquet(output_path)
                self.logger.info(f"Saved df_decursus vg to {output_path}")

            return df_decursus

        except Exception as e:
            self.logger.error(f"Error in vg filter on decursus data: {e}")
            return None

    @timer_log_method(logger_attr="logger")
    def apply_filter(self, df_decursus):
        try:
            df_filtered = df_decursus.loc[
                (
                    (df_decursus[self.vg_indication] == False)  # noqa: E712
                    | (df_decursus[self.vg_indication].isna())
                )
                & (df_decursus[self.pob_indication]),
                [
                    self.studyId,
                    self.index_date_input_col,
                    self.consult_id,
                    self.section_title,
                    self.section_text,
                    self.pob_indication,
                    self.vg_indication,
                ],
            ]

            # print dtype before conversion
            self.logger.info(f"Data type of {self.studyId} before conversion: {df_filtered[self.studyId].dtype}")

            df_filtered = self.ensure_string_column(df_filtered, self.studyId)

            df_filtered = df_filtered.rename(columns={self.index_date_input_col: self.index_date})

            self.logger.info(f"Number of patients in patsel_pob_non_vg: {df_filtered[self.studyId].nunique()}")
            self.logger.info(f"Number of entries in patsel_pob_non_vg: {len(df_filtered)}")

            if self.save_intermediates:
                folder = f"intermediates/{self.mode}"
                output_path = Path(get_project_root()) / "data" / folder / "patsel_pob_non_vg.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_filtered.to_parquet(output_path)
                self.logger.info(f"Saved df_filtered to {output_path}")

            return df_filtered

        except Exception as e:
            self.logger.error(f"Error in applying filter on decursus data: {e}")
            return None

    @timer_log_method(logger_attr="logger")
    def get_filtered_patsel(self, dfPatsel, df_filtered):
        try:
            dfPatsel_final = dfPatsel.merge(
                df_filtered[
                    [
                        self.studyId,
                        self.index_date,
                        self.pob_indication,
                        self.vg_indication,
                    ]
                ],
                how="inner",
                on=[self.studyId, self.index_date],
            )
            dfPatsel_final = dfPatsel_final.drop_duplicates()

            self.logger.info(f"Number of patients in dfPatsel after merge: {dfPatsel_final[self.studyId].nunique()}")
            self.logger.info(f"Number of entries in dfPatsel after merge: {len(dfPatsel_final)}")

            if self.save_intermediates:
                folder = f"intermediates/{self.mode}"
                output_path = Path(get_project_root()) / "data" / folder / "patsel_filtered.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                dfPatsel_final.to_parquet(output_path)
                self.logger.info(f"Saved filtered patsel to {output_path}")

            return dfPatsel_final

        except Exception as e:
            self.logger.error(f"Error in merging patsel with pob and vg filter: {e}")
            return None

    @timer_log_method(logger_attr="logger")
    def main(self):
        # Step 1: Loading and processing patsel data using PatientSelector
        if self.example:
            dfPatsel = pd.read_parquet(Path(get_project_root())/"data"/"example_data"/"patsel_raw.parquet")
        else:
            dfPatsel = self.load_patsel_data()

        # Initialize variables for tracking
        df_decursus = None
        df_pob = None
        df_pob_vg = None
        df_filtered = None

        # Step 2 + 3. Load decursus data and applying POB and VG filters on decursus data  # noqa: E501
        if self.save_intermediates:
            folder = f"intermediates/{self.mode}"

            # Path to patsel with pob and vg filter (complete filtering)
            output_path_filtered = Path(get_project_root()) / "data" / folder / "patsel_pob_non_vg.parquet"

            # Path to patsel with pob and vg indicators (not filtered yet!)
            output_path_vg = Path(get_project_root()) / "data" / folder / "patsel_pob_vg_indicators.parquet"

            # Path to pob indicator data (no VG indicator yet)
            output_path_pob = Path(get_project_root()) / "data" / folder / "patsel_pob.parquet"

            if output_path_filtered.exists():
                self.logger.info(f"Loading existing filtered data from {output_path_filtered}")
                df_filtered = pd.read_parquet(output_path_filtered)

            elif output_path_vg.exists():
                self.logger.info(f"Loading existing dataframe with pob and vg indicators from {output_path_vg}")
                df_pob_vg = pd.read_parquet(output_path_vg)
                df_filtered = self.apply_filter(df_pob_vg)

            elif output_path_pob.exists():
                self.logger.info(f"Loading existing dataframe with pob indicator from {output_path_pob}")
                df_pob = pd.read_parquet(output_path_pob)
                df_pob_vg = self.vg_filter(df_pob)
                df_filtered = self.apply_filter(df_pob_vg)

            else:
                df_filtered = None

        # If we haven't loaded data yet (either save_intermediates is False or no
        # files existed)
        if df_filtered is None:
            self.logger.info("Load all decursus data from scratch as no existing filtered data was found")

            # Step 2. Loading decursus (consult) data
            if self.example:
                df_decursus = pd.read_parquet(Path(get_project_root())/"data"/"example_data"/"decursus_raw.parquet")
            else:
                df_decursus = self.load_decursus_data()

            self.logger.info(f"Number of unique patients in consult database: {df_decursus[self.studyId].nunique()}")

            # keep only the rows for studyIDs that are present in patsel
            df_decursus = self.ensure_string_column(df_decursus, self.studyId)
            dfPatsel = self.ensure_string_column(dfPatsel, self.studyId)
            df_decursus = df_decursus.loc[df_decursus[self.studyId].isin(dfPatsel[self.studyId].unique())]
            self.logger.info(
                f"Number of unique patients from patsel with decursus data: {df_decursus[self.studyId].nunique()}"
            )

            # Step 3. Applying POB and VG filters on decursus data
            df_pob = self.filter_pob(df_decursus)
            df_pob_vg = self.vg_filter(df_pob)
            df_filtered = self.apply_filter(df_pob_vg)
            self.logger.info(
                f"Number of unique patients with chest pain and no cardio history: {df_filtered[self.studyId].nunique()}"
            )

            if df_filtered is None:
                self.logger.error("apply_filter() returned None")
                return None

        # Step 4. Merging filtered data with patsel data
        dfPatsel_final = self.get_filtered_patsel(dfPatsel, df_filtered)
        dfPatsel_final = dfPatsel_final.drop_duplicates()
        # Drop duplicated Identification numbers:
        dfPatsel_final = dfPatsel_final.drop_duplicates(subset=[self.studyId], keep="first")

        self.logger.info(f"Number of unique patients in final patsel: {dfPatsel_final[self.studyId].nunique()}")

        if dfPatsel_final is None:
            self.logger.error("get_filtered_patsel() returned None. No patient selection.")
            return None

        # Return all results and statistics
        results = {
            "dfPatsel": dfPatsel,
            "dfPatsel_final": dfPatsel_final,
            "df_pob_vg": df_pob_vg,
            "df_filtered": df_filtered,
            "stats": {
                "initial_unique_patients": dfPatsel[self.studyId].nunique(),
                "filtered_count": len(df_filtered),
                "final_count": len(dfPatsel_final),
                "unique_patients": dfPatsel_final[self.studyId].nunique(),
                "retention_rate": dfPatsel_final[self.studyId].nunique() / dfPatsel[self.studyId].nunique() * 100,
            },
        }

        self.logger.info(f"ETL workflow completed. Final patient count: {len(dfPatsel_final):,}, {results['stats']}")

        return results


# # ### Full main function ######
# if __name__ == "__main__":
#     logging_enabled = True
#     version = "v2_ARGUS25"
#     mode="inference"
#     debug_mode = True

#     # TODO change log option capable for posit
#     logger = Logger(
#         log_file=get_logs_dir() / "ETL_patsel_filter.log",
#         enabled=logging_enabled,
#     )

#     ### Test train #####
#     # path_manager = PathManager_RES_Folder_UPOD(logger)

#     # processor = PatselProcessor(
#     #         logger=logger,
#     #         mode="train",
#     #         version="v2_ARGUS25",
#     #         save_intermediates=True,
#     #         path_manager=path_manager,
#     #     )

#     # processor.main()

#     #### Test Inference - currently only load_patsel
#     processor = PatselProcessor(
#         logger=logger,
#         mode=mode,
#         version=version,
#         debug_mode=debug_mode #False is not specified
#     )

#     processor.main()

# # SELECT id, section_text, section_title_display_original
# #                 FROM [SELF_SERVICE].[argus].[Consult]
# #                 OPTION (MAXDOP 1);
