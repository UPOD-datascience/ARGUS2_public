import logging
import os
import tomllib
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import uvicorn
from argus.Pipeline.Run import RunInference
from argus.utils.auth import check_authorisation
from argus.utils.db_utils import DatabaseUtils
from argus.utils.general_functions import get_logs_dir
from argus.utils.logger import Logger
from argus.utils.PathManager import PathManager_RES_Folder_UPOD
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.security.api_key import APIKeyHeader

logging_enabled = True

logger = Logger(
    log_file=get_logs_dir() / "RunInference.log",
    enabled=True,
    df_level=logging.DEBUG,
)

load_dotenv()

app = FastAPI()
header_scheme = APIKeyHeader(name="X-API-KEY", auto_error=False)

with open(Path(__file__).parents[3] / "pyproject.toml", "rb") as f:
    config = tomllib.load(f)

API_VERSION = config["project"]["version"]

logger = Logger(log_file=get_logs_dir() / "RunInference.log", enabled=True)
path_manager = PathManager_RES_Folder_UPOD(logger)


# Job status tracking
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Store job statuses in memory (consider using Redis for production)
job_statuses: Dict[str, Dict[str, Any]] = {}

# Run Inference
processor = RunInference(mode="inference", logger=logger)


def get_last_index_from_db(
    table_name: str = "shadow_test", date_column: str = "Index_dt"
):
    schema = os.getenv("DB_POSIT_SCHEMA", "argus")
    db_utils = DatabaseUtils(logger=logger)
    return db_utils.last_index(table_name, date_column, schema)


def write_to_posit_connect(df_female: pd.DataFrame, df_male: pd.DataFrame):
    """
    Write the inference results to the Posit Connect database.

    Args:
        df_female: DataFrame containing female patient predictions
        df_male: DataFrame containing male patient predictions
    """
    try:
        # Initialize database utilities
        db_utils = DatabaseUtils(logger=logger)

        # Get database schema from environment
        db_schema = os.getenv("DB_POSIT_SCHEMA", "argus")

        # Add gender column to distinguish between male and female dataframes
        if not df_female.empty:
            df_female = df_female.copy()
            if "gender" not in df_female.columns:
                df_female["gender"] = "F"

        if not df_male.empty:
            df_male = df_male.copy()
            if "gender" not in df_male.columns:
                df_male["gender"] = "M"

        # Combine dataframes
        combined_df = pd.concat([df_female, df_male], ignore_index=True)
        # output a dictionary with the columns names and their types
        # output a dictionary with the columns names and their types
        output_dict = {col: str(combined_df[col].dtype) for col in combined_df.columns}
        logger.info(f"Output dictionary: {output_dict}")

        # Validate the combined dataframe
        is_valid, validation_issues = db_utils.validate_dataframe_schema(combined_df)
        if not is_valid:
            logger.warning(f"Validation issues found: {validation_issues}")

        # Write to database using the utility function
        # This will automatically handle data preparation, metadata addition, and robust connection
        success = db_utils.write_dataframe_to_db(
            df=combined_df,
            table_name="shadow_test",
            schema=db_schema,
            if_exists="append",
            validate=True,
            prepare=True,
            add_metadata=True,
            chunksize=1000,
        )

        if success:
            logger.info(
                f"Successfully wrote {len(combined_df)} rows to {db_schema}.shadow_test"
            )

        return success

    except Exception as e:
        logger.error(f"Error writing predictions to Posit Connect database: {str(e)}")
        raise


def write_logging_to_posit_connect(df_log: pd.DataFrame):
    """
    Write the logging to the Posit Connect database.
    """
    try:
        # Initialize database utilities
        db_utils = DatabaseUtils(logger=logger)

        # Get database schema from environment
        db_schema = os.getenv("DB_POSIT_SCHEMA", "argus")

        # Write to database using the utility function
        # This will automatically handle data preparation, metadata addition, and robust connection
        success = db_utils.write_dataframe_to_db(
            df=df_log,
            table_name="logging_shadow_test",
            schema=db_schema,
            if_exists="append",
            validate=True,
            prepare=True,
            add_metadata=True,
            chunksize=1000,
        )

        if success:
            logger.info(
                f"Successfully wrote {len(df_log)} rows to {db_schema}.logging_shadow_test"
            )

        return success

    except Exception as e:
        logger.error(f"Error writing logging to Posit Connect database: {str(e)}")
        raise


def run_inference_task(job_id: str, last_date: Optional[datetime] = None):
    """Background task to run inference"""
    try:
        # Update status to running
        job_statuses[job_id]["status"] = JobStatus.RUNNING
        job_statuses[job_id]["started_at"] = datetime.now().isoformat()

        # Run the actual inference
        df_female, df_male = processor.run(from_date=last_date)

        # Write results to Posit Connect database
        write_to_posit_connect(df_female, df_male)

        # Update status to completed
        job_statuses[job_id]["status"] = JobStatus.COMPLETED
        job_statuses[job_id]["completed_at"] = datetime.now().isoformat()
        job_statuses[job_id]["result"] = {
            "success": True,
            "message": "Inference completed successfully",
        }

    except Exception as e:
        logger.debug(f"Error running processor: {e}")
        job_statuses[job_id]["status"] = JobStatus.FAILED
        job_statuses[job_id]["completed_at"] = datetime.now().isoformat()
        job_statuses[job_id]["error"] = str(e)

    finally:
        df_log = logger.to_dataframe()
        write_logging_to_posit_connect(df_log)


@app.get("/get_last_index")
def get_last_index():
    last_index = get_last_index_from_db()
    return {"last_index": last_index}


@app.post("/run_from_last")
async def run_from_last(
    background_tasks: BackgroundTasks, api_key: str = Depends(header_scheme)
):
    """
    Start inference job in the background and return job ID immediately.
    This prevents timeout for long-running tasks.
    """

    check_authorisation(api_key, "X_API_KEY_post")

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Initialize job status
    job_statuses[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }

    last_date = get_last_index_from_db()

    logger.info(f"Starting inference starting from: {last_date}")
    # Add task to background
    background_tasks.add_task(run_inference_task, job_id, last_date=last_date)

    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "last_date": last_date,
        "message": "Inference job started. Use /job_status/{job_id} to check progress.",
    }


@app.post("/run_all")
async def run_all(
    background_tasks: BackgroundTasks, api_key: str = Depends(header_scheme)
):
    """
    Start inference job in the background and return job ID immediately.
    This prevents timeout for long-running tasks.
    """

    check_authorisation(api_key, "X_API_KEY_post")

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Initialize job status
    job_statuses[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    # Add task to background
    background_tasks.add_task(run_inference_task, job_id, last_date=None)

    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "message": "Inference job started. Use /job_status/{job_id} to check progress.",
    }


@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """
    Check the status of a running job
    """
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_statuses[job_id]


@app.get("/jobs")
async def list_jobs(limit: int = 10, status: Optional[JobStatus] = None):
    """
    List recent jobs with optional status filter
    """
    jobs = list(job_statuses.values())

    # Filter by status if provided
    if status:
        jobs = [j for j in jobs if j["status"] == status]

    # Sort by creation date
    sorted_jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)[:limit]

    return {
        "total": len(job_statuses),
        "filtered": len(sorted_jobs),
        "jobs": sorted_jobs,
    }


@app.delete("/job/{job_id}")
async def clear_job(job_id: str):
    """
    Clear a completed job from memory
    """
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_statuses[job_id]
    if job["status"] in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400, detail="Cannot delete job that is still running"
        )

    del job_statuses[job_id]
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/shadow_test/latest")
async def get_latest_shadow_test_records(limit: int = 100):
    """
    Get the latest records from shadow_test table

    Args:
        limit: Maximum number of records to return (default: 100)

    Returns:
        Dictionary containing the records and metadata
    """
    try:
        db_utils = DatabaseUtils(logger=logger)
        db_schema = os.getenv("DB_POSIT_SCHEMA", "argus")

        query = f"""
        SELECT TOP ({limit}) *
        FROM {db_schema}.shadow_test
        ORDER BY created_at DESC
        """

        df = db_utils.read_from_db(query)

        # Convert datetime columns to ISO format strings for JSON serialization
        datetime_columns = df.select_dtypes(include=["datetime64"]).columns
        for col in datetime_columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "count": len(df),
            "total_limit": limit,
            "records": df.to_dict(orient="records"),
        }

    except Exception as e:
        logger.error(f"Error querying shadow_test table: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.get("/shadow_test/stats")
async def get_shadow_test_stats():
    """
    Get statistics about the shadow_test table

    Returns:
        Dictionary containing table statistics
    """
    try:
        db_utils = DatabaseUtils(logger=logger)
        db_schema = os.getenv("DB_POSIT_SCHEMA", "argus")

        # Query to get statistics
        stats_query = f"""
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT run_id) as unique_runs,
            COUNT(DISTINCT study_id) as unique_studies,
            MIN(created_at) as earliest_record,
            MAX(created_at) as latest_record,
            SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) as male_count,
            SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) as female_count,
            AVG(predicted_risk) as avg_predicted_risk,
            AVG(missing_fraction) as avg_missing_fraction
        FROM {db_schema}.shadow_test
        """

        df = db_utils.read_from_db(stats_query)

        if not df.empty:
            stats = df.iloc[0].to_dict()

            # Convert datetime to string if present
            for key in ["earliest_record", "latest_record"]:
                if key in stats and pd.notna(stats[key]):
                    stats[key] = pd.to_datetime(stats[key]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

            # Convert numpy types to Python types for JSON serialization
            for key, value in stats.items():
                if pd.isna(value):
                    stats[key] = None
                elif hasattr(value, "item"):  # numpy type
                    stats[key] = value.item()

            return {"statistics": stats}
        else:
            return {"statistics": {"total_records": 0}}

    except Exception as e:
        logger.error(f"Error getting shadow_test statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Statistics query failed: {str(e)}"
        )


@app.get("/shadow_test/by_run/{run_id}")
async def get_shadow_test_by_run(run_id: str):
    """
    Get records from shadow_test table for a specific run_id

    Args:
        run_id: The run_id to filter by

    Returns:
        Dictionary containing the records for the specified run
    """
    try:
        db_utils = DatabaseUtils(logger=logger)
        db_schema = os.getenv("DB_POSIT_SCHEMA", "argus")

        query = f"""
        SELECT *
        FROM {db_schema}.shadow_test
        WHERE run_id = :run_id
        ORDER BY study_id
        """

        df = db_utils.read_from_db(query, params={"run_id": run_id})

        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"No records found for run_id: {run_id}"
            )

        # Convert datetime columns to ISO format strings for JSON serialization
        datetime_columns = df.select_dtypes(include=["datetime64"]).columns
        for col in datetime_columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "run_id": run_id,
            "count": len(df),
            "records": df.to_dict(orient="records"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying shadow_test by run_id: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "version": API_VERSION,
        "active_jobs": sum(
            1
            for j in job_statuses.values()
            if j["status"] in [JobStatus.PENDING, JobStatus.RUNNING]
        ),
    }


if __name__ == "__main__":
    uvicorn.run(app=app)
