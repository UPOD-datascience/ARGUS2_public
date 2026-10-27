import pandas as pd
from pydantic import BaseModel, field_validator, model_validator


class PatselDataSchema(BaseModel):
    """Schema for validating the complete patsel dataset output"""

    data: pd.DataFrame
    record_count: int
    columns: list[str]
    version: str = "v2_ARGUS25"
    mode: str = "train"

    class Config:
        arbitrary_types_allowed = True  # allow for dataframe check

    @model_validator(mode="after")  # Runs AFTER all fields are set
    def validate_dataframe(self) -> "PatselDataSchema":
        """Validate DataFrame structure and types"""
        df = self.data
        version = self.version
        mode = self.mode

        # Load config to get column names
        from argus.utils.config import load_config

        config = load_config("general", version=version, mode=mode)
        cols = config["columns"]

        # Define expected types using config column names
        expected_types = {
            cols["study_id"]: "object",
            cols["gender"]: ["object", "int64"],  # Can be either
            cols["age"]: "int64",
            cols["index_date"]: "datetime64[ns]",
        }

        # Check all expected columns exist
        missing = set(expected_types.keys()) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Check data types
        for col, expected in expected_types.items():
            actual = str(df[col].dtype)
            expected_list = [expected] if isinstance(expected, str) else expected
            if actual not in expected_list:
                raise ValueError(f"Column '{col}' has type '{actual}', expected {expected_list}")

        # Basic checks
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Validate age range
        age_col = cols["age"]
        if (df[age_col] < 18).any():
            invalid_count = (df[age_col] < 18).sum()
            raise ValueError(f"Age below 18 found in {invalid_count} records")

        if (df[age_col] > 120).any():
            invalid_count = (df[age_col] > 120).sum()
            raise ValueError(f"Age above 120 found in {invalid_count} records")

        return df

    @field_validator("record_count")
    @classmethod
    def validate_record_count(cls, v: int, info) -> int:
        """Ensure record count matches DataFrame length"""
        df = info.data.get("data")
        if df is not None and len(df) != v:
            raise ValueError(f"Record count mismatch: {v} != {len(df)}")
        return v

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, version: str = "v2_ARGUS25", mode: str = "train"):
        """Create schema from DataFrame"""
        return cls(data=df, record_count=len(df), columns=list(df.columns), version=version, mode=mode)
