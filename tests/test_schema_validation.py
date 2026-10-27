#!/usr/bin/env python3
"""Test script to validate dataframe schema with the updated ModelOutput structure."""

import logging

# Add parent directory to path for imports
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from argus.data_models.model_output import ModelOutput, get_expected_dtypes
from argus.utils.db_utils import DatabaseUtils

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_sample_dataframe():
    """Create a sample dataframe with the expected structure."""
    # Sample data matching the provided schema
    data = {
        "patientPseuID": ["P001", "P002", "P003"],
        "gender": ["M", "F", "M"],
        "Index_dt": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "Age": [45, 67, 55],
        "BMI": [25.3, 28.1, 22.5],
        "DiastolicBP": [80.0, 85.0, 78.0],
        "SystolicBP": [120.0, 130.0, 118.0],
        "afname_dt": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-02 11:00:00", "2024-01-03 09:00:00"]),
        "ALAT": [25.0, 30.0, 22.0],
        "ASAT": [28.0, 32.0, 25.0],
        "CRP": [2.5, 5.0, 1.8],
        "Cholesterol": [180.0, 210.0, 165.0],
        "GGT": [35.0, 42.0, 30.0],
        "Glucose": [95.0, 110.0, 88.0],
        "HDL Cholesterol": [55.0, 45.0, 60.0],
        "Hemoglobine": [14.5, 13.2, 15.1],
        "Kalium": [4.2, 4.0, 4.3],
        "Kreatinine": [0.9, 1.1, 0.8],
        "LD": [150.0, 180.0, 140.0],
        "LDL Cholesterol": [110.0, 140.0, 95.0],
        "Leucocyten": [6.5, 7.8, 6.0],
        "MCH": [29.0, 28.5, 29.5],
        "MCHC": [33.5, 33.0, 34.0],
        "MCV": [88.0, 86.0, 87.0],
        "Natrium": [140.0, 138.0, 141.0],
        "TSH": [2.1, 3.5, 1.8],
        "Triglyceriden": [120.0, 180.0, 95.0],
        "Trombocyten": [250.0, 280.0, 230.0],
        "eGFR (CKD-EPI)": [90.0, 75.0, 95.0],
        "InjClsInfarction": [0.0, 0.0, 1.0],
        "ST_deviation_index_Anterior": [0.5, 0.8, 1.2],
        "ST_deviation_index_Inferior": [0.3, 0.6, 0.9],
        "ST_deviation_index_Lateral": [0.4, 0.7, 1.0],
        "T_morphology_score_Anterior": [0.2, 0.4, 0.6],
        "T_morphology_score_Inferior": [0.1, 0.3, 0.5],
        "T_morphology_score_Lateral": [0.15, 0.35, 0.55],
        "Regional_ischemia_Anterior": [0.0, 0.0, 1.0],
        "Regional_ischemia_Inferior": [0.0, 1.0, 0.0],
        "Regional_ischemia_Lateral": [0.0, 0.0, 0.0],
        "predicted_risk": [0.15, 0.45, 0.65],
        "missing_fraction": [0.05, 0.10, 0.02],
    }

    return pd.DataFrame(data)


def create_dataframe_with_old_names():
    """Create a sample dataframe with the old column names."""
    data = {
        "study_id": ["P001", "P002"],
        "gender": ["M", "F"],
        "age": [45, 67],
        "index_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "meting_bmi": [25.3, 28.1],
        "meting_bp_dia": [80.0, 85.0],
        "meting_bp_syst": [120.0, 130.0],
        "HDL_Cholesterol": [55.0, 45.0],
        "LDL_Cholesterol": [110.0, 140.0],
        "eGFR_CKD_EPI": [90.0, 75.0],
    }

    return pd.DataFrame(data)


def test_schema_validation():
    """Test the schema validation functionality."""
    logger.info("Starting schema validation tests...")

    # Initialize utilities
    db_utils = DatabaseUtils(logger=logger)

    # Test 1: Validate correct schema
    logger.info("\nTest 1: Validating dataframe with correct schema...")
    df_correct = create_sample_dataframe()
    is_valid, issues = db_utils.validate_dataframe_schema(df_correct)

    if is_valid:
        logger.info("✓ Validation passed for correct schema")
    else:
        logger.error(f"✗ Validation failed: {issues}")

    # Test 2: Validate with missing required columns
    logger.info("\nTest 2: Validating dataframe with missing required columns...")
    df_missing = df_correct.drop(columns=["patientPseuID"])
    is_valid, issues = db_utils.validate_dataframe_schema(df_missing)

    if not is_valid and "Missing required columns" in str(issues):
        logger.info(f"✓ Correctly identified missing columns: {issues}")
    else:
        logger.error("✗ Failed to identify missing required columns")

    # Test 3: Test column migration
    logger.info("\nTest 3: Testing column name migration...")
    df_old_names = create_dataframe_with_old_names()
    df_migrated = db_utils.migrate_column_names(df_old_names)

    # Check if migration worked
    expected_new_columns = ["patientPseuID", "Age", "Index_dt", "BMI", "DiastolicBP", "SystolicBP"]
    migrated_success = all(col in df_migrated.columns for col in expected_new_columns)

    if migrated_success:
        logger.info("✓ Column migration successful")
        logger.info(f"  Old columns: {list(df_old_names.columns)}")
        logger.info(f"  New columns: {list(df_migrated.columns)}")
    else:
        logger.error("✗ Column migration failed")

    # Test 4: Test reverse migration
    logger.info("\nTest 4: Testing reverse column migration...")
    df_reversed = db_utils.migrate_column_names(df_migrated, reverse=True)

    reverse_success = all(col in df_reversed.columns for col in ["study_id", "age", "index_date"])
    if reverse_success:
        logger.info("✓ Reverse migration successful")
    else:
        logger.error("✗ Reverse migration failed")

    # Test 5: Test dataframe preparation for DB
    logger.info("\nTest 5: Testing dataframe preparation for database...")
    df_prepared = db_utils.prepare_dataframe_for_db(df_correct, add_metadata=True)

    # Check if metadata columns were added
    metadata_added = "created_at" in df_prepared.columns and "run_id" in df_prepared.columns
    if metadata_added:
        logger.info("✓ Dataframe preparation successful with metadata")
        logger.info("  Added columns: created_at, run_id")
    else:
        logger.error("✗ Failed to add metadata columns")

    # Test 6: Validate ModelOutput with sample data
    logger.info("\nTest 6: Testing ModelOutput model with sample data...")
    try:
        # Take first row and convert to dict
        sample_row = df_correct.iloc[0].to_dict()
        model_instance = ModelOutput(**sample_row)
        logger.info("✓ ModelOutput successfully created from dataframe row")
        logger.info(f"  Patient ID: {model_instance.patientPseuID}")
        logger.info(f"  Age: {model_instance.Age}")
        logger.info(f"  BMI: {model_instance.BMI}")
    except Exception as e:
        logger.error(f"✗ Failed to create ModelOutput: {str(e)}")

    logger.info("\n" + "=" * 50)
    logger.info("Schema validation tests completed!")


def display_expected_schema():
    """Display the expected schema for reference."""
    logger.info("\nExpected DataFrame Schema:")
    logger.info("=" * 50)

    expected_dtypes = get_expected_dtypes()

    # Group by data type
    dtype_groups = {}
    for col, dtype in expected_dtypes.items():
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append(col)

    # Display grouped by type
    for dtype, columns in sorted(dtype_groups.items()):
        logger.info(f"\n{dtype}:")
        for col in sorted(columns):
            logger.info(f"  - {col}")


if __name__ == "__main__":
    display_expected_schema()
    test_schema_validation()
