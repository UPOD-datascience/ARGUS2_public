from datetime import datetime

import pandas as pd
from pydantic import BaseModel, field_validator


class PatselModel(BaseModel):
    """Pydantic model for Patsel data"""

    studyId_0888: float
    gender: str  # not mapped yet
    Age: int
    index_dt: datetime

    @field_validator("studyId_0888", mode="before")
    @classmethod
    def check_not_nan(cls, v):
        if pd.isna(v):
            raise ValueError("Value cannot be NaN")
        return v


# class LabModel(BaseModel):
