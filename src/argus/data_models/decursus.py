import pandas as pd
from pydantic import BaseModel, model_validator


class DecursusDataSchema(BaseModel):
    """Schema for validating decursus dataset"""

    data: pd.DataFrame
    record_count: int
    columns: list[str]
    version: str = "v2_ARGUS25"
    mode: str = "train"

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_dataframe(self) -> "DecursusDataSchema":
        """Validate DataFrame structure"""
        df = self.data

        # Load patsel_processor config to get required columns
        from argus.utils.config import load_config

        general_config = load_config("general", version=self.version, mode=self.mode)
        processor_config = load_config("patsel_processor", version=self.version, mode=self.mode)

        # Get column names
        study_id_col = general_config["columns"]["study_id"]
        cols = processor_config.get("columns", {})

        required_columns = [cols.get("consult_id_col"), cols.get("section_text_col"), cols.get("section_title_col")]

        # Remove None values and add study_id if present
        required_columns = [col for col in required_columns if col is not None]
        if study_id_col in df.columns:
            required_columns.append(study_id_col)

        # Check required columns exist
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Validate studyId is string type (to match patsel)
        if study_id_col in df.columns:
            if df[study_id_col].dtype != "object":
                raise ValueError(
                    f"{study_id_col} must be string type (object), "
                    f"got {df[study_id_col].dtype}. "
                    f"This must match patsel data type for merging."
                )

        # Basic checks
        if df.empty:
            raise ValueError("DataFrame is empty")

        return self

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, version: str = "v2_ARGUS25", mode: str = "train"):
        """Create schema from DataFrame"""
        return cls(data=df, record_count=len(df), columns=list(df.columns), version=version, mode=mode)
