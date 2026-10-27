from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, Field


class ModelOutput(BaseModel):
    """Pydantic model representing the structure of the ARGUS model output.
    This model defines the expected columns and their types for the output dataframes.
    """

    # Basic patient information
    patientPseuID: str = Field(..., description="Patient pseudo identifier")
    gender: Union[str, int] = Field(..., description="Patient gender")
    Age: int = Field(..., description="Patient age")
    Index_dt: datetime = Field(..., description="Index date of the study")

    # Body measurements
    BMI: Optional[float] = Field(None, description="Body Mass Index")
    DiastolicBP: Optional[float] = Field(None, description="Diastolic blood pressure")
    SystolicBP: Optional[float] = Field(None, description="Systolic blood pressure")

    # Lab test timestamp
    afname_dt: Optional[datetime] = Field(None, description="Lab test collection datetime")

    # Liver enzymes
    ALAT: Optional[float] = Field(None, description="Alanine aminotransferase")
    ASAT: Optional[float] = Field(None, description="Aspartate aminotransferase")
    GGT: Optional[float] = Field(None, description="Gamma-glutamyl transferase")
    LD: Optional[float] = Field(None, description="Lactate dehydrogenase")

    # Inflammatory markers
    CRP: Optional[float] = Field(None, description="C-reactive protein")

    # Lipid panel
    Cholesterol: Optional[float] = Field(None, description="Total cholesterol")
    HDL_Cholesterol: Optional[float] = Field(None, alias="HDL Cholesterol", description="HDL cholesterol")
    LDL_Cholesterol: Optional[float] = Field(None, alias="LDL Cholesterol", description="LDL cholesterol")
    Triglyceriden: Optional[float] = Field(None, description="Triglycerides")

    # Blood sugar
    Glucose: Optional[float] = Field(None, description="Blood glucose level")

    # Blood count
    Hemoglobine: Optional[float] = Field(None, description="Hemoglobin")
    Leucocyten: Optional[float] = Field(None, description="Leukocytes (white blood cells)")
    Trombocyten: Optional[float] = Field(None, description="Thrombocytes (platelets)")
    MCH: Optional[float] = Field(None, description="Mean corpuscular hemoglobin")
    MCHC: Optional[float] = Field(None, description="Mean corpuscular hemoglobin concentration")
    MCV: Optional[float] = Field(None, description="Mean corpuscular volume")

    # Electrolytes
    Kalium: Optional[float] = Field(None, description="Potassium")
    Natrium: Optional[float] = Field(None, description="Sodium")

    # Kidney function
    Kreatinine: Optional[float] = Field(None, description="Creatinine")
    eGFR_CKD_EPI: Optional[float] = Field(
        None, alias="eGFR (CKD-EPI)", description="Estimated glomerular filtration rate"
    )

    # Thyroid
    TSH: Optional[float] = Field(None, description="Thyroid stimulating hormone")

    # ECG/Cardiac markers
    InjClsInfarction: Optional[float] = Field(None, description="Infarction injury classification")
    ST_deviation_index_Anterior: Optional[float] = Field(None, description="ST deviation index - Anterior")
    ST_deviation_index_Inferior: Optional[float] = Field(None, description="ST deviation index - Inferior")
    ST_deviation_index_Lateral: Optional[float] = Field(None, description="ST deviation index - Lateral")
    T_morphology_score_Anterior: Optional[float] = Field(None, description="T wave morphology score - Anterior")
    T_morphology_score_Inferior: Optional[float] = Field(None, description="T wave morphology score - Inferior")
    T_morphology_score_Lateral: Optional[float] = Field(None, description="T wave morphology score - Lateral")
    Regional_ischemia_Anterior: Optional[float] = Field(None, description="Regional ischemia - Anterior")
    Regional_ischemia_Inferior: Optional[float] = Field(None, description="Regional ischemia - Inferior")
    Regional_ischemia_Lateral: Optional[float] = Field(None, description="Regional ischemia - Lateral")

    # Model predictions
    predicted_risk: Optional[float] = Field(None, description="Predicted risk score")
    missing_fraction: Optional[float] = Field(None, description="Fraction of missing data")

    class Config:
        # Allow population by field name for columns with spaces
        populate_by_name = True
        # Allow extra fields in case there are additional columns
        extra = "allow"
        # Handle datetime serialization
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


def get_expected_dtypes() -> dict:
    """Returns a dictionary of expected column names and their pandas dtypes.
    This can be used to validate or cast dataframe columns.
    """
    return {
        "patientPseuID": "object",
        "gender": "object",  # Can be cast to int64 if needed
        "Age": "int64",
        "Index_dt": "datetime64[ns]",
        "BMI": "float64",
        "DiastolicBP": "float64",
        "SystolicBP": "float64",
        "afname_dt": "datetime64[ns]",
        "ALAT": "float64",
        "ASAT": "float64",
        "CRP": "float64",
        "Cholesterol": "float64",
        "GGT": "float64",
        "Glucose": "float64",
        "HDL Cholesterol": "float64",
        "Hemoglobine": "float64",
        "Kalium": "float64",
        "Kreatinine": "float64",
        "LD": "float64",
        "LDL Cholesterol": "float64",
        "Leucocyten": "float64",
        "MCH": "float64",
        "MCHC": "float64",
        "MCV": "float64",
        "Natrium": "float64",
        "TSH": "float64",
        "Triglyceriden": "float64",
        "Trombocyten": "float64",
        "eGFR (CKD-EPI)": "float64",
        "InjClsInfarction": "float64",
        "ST_deviation_index_Anterior": "float64",
        "ST_deviation_index_Inferior": "float64",
        "ST_deviation_index_Lateral": "float64",
        "T_morphology_score_Anterior": "float64",
        "T_morphology_score_Inferior": "float64",
        "T_morphology_score_Lateral": "float64",
        "Regional_ischemia_Anterior": "float64",
        "Regional_ischemia_Inferior": "float64",
        "Regional_ischemia_Lateral": "float64",
        "predicted_risk": "float64",
        "missing_fraction": "float64",
    }


def get_column_mappings() -> dict:
    """Returns a dictionary mapping old column names to new column names.
    This can be used for backward compatibility or data migration.
    """
    return {
        "study_id": "patientPseuID",
        "age": "Age",
        "index_date": "Index_dt",
        "meting_bmi": "BMI",
        "meting_bp_dia": "DiastolicBP",
        "meting_bp_syst": "SystolicBP",
        # Special columns with spaces remain the same
        "HDL_Cholesterol": "HDL Cholesterol",
        "LDL_Cholesterol": "LDL Cholesterol",
        "eGFR_CKD_EPI": "eGFR (CKD-EPI)",
    }
