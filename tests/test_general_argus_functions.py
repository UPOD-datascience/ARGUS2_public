import os
import sys
import unittest
from unittest.mock import MagicMock

import pandas as pd
from src.utils.general_argus_functions import DataProcessingUtils

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


class TestDataProcessingUtils(unittest.TestCase):
    """Test cases for the DataProcessingUtils class."""

    def setUp(self):
        """Set up test fixtures, called before every test method."""
        # Create a mock logger
        self.mock_logger = MagicMock()
        self.mock_logger.error = MagicMock()
        self.mock_logger.warning = MagicMock()
        self.mock_logger.debug = MagicMock()

        # Create a sample dataframe
        self.sample_df = pd.DataFrame(
            {
                "studyId_0771": ["P001", "P002", "P003", "P004", "P005"],
                "gender": ["male", "female", "male", "unknown", None],
                "Age": [25, 45, 17, 120, 65],
                "CT_Date": [
                    "2023-01-15",
                    "2023-02-20",
                    "2023-03-10",
                    "invalid_date",
                    "2023-05-05",
                ],
                "dob": [
                    "1998-01-01",
                    "1978-01-01",
                    "2006-01-01",
                    "1903-01-01",
                    "1958-01-01",
                ],
                "deceasedDateTime": [None, "2023-12-31", None, None, "2023-11-15"],
            }
        )

    def test_validate_input_data_valid(self):
        """Test validation with valid input data."""
        required_cols = ["studyId_0771", "gender", "Age"]

        # Should not raise an exception
        result = DataProcessingUtils.validate_input_data(self.sample_df, required_cols, logger=self.mock_logger)

        self.assertTrue(result)
        self.mock_logger.error.assert_not_called()

    def test_validate_input_data_not_dataframe(self):
        """Test validation with non-DataFrame input."""
        required_cols = ["studyId_0771", "gender", "Age"]
        invalid_df = {"studyId_0771": ["P001"], "gender": ["male"], "Age": [25]}

        with self.assertRaises(TypeError):
            DataProcessingUtils.validate_input_data(
                invalid_df,
                required_cols,
                id_col="studyId_0771",
                logger=self.mock_logger,
            )

        self.mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
