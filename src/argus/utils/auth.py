import os

from fastapi import HTTPException

from argus.utils.logger import Logger

logger = Logger(__name__)


def check_authorisation(key: str, stored_key: str) -> None:
    # Check if the provided key matches the stored key in the environment variables.
    if key != os.environ[stored_key]:
        logger.error("403: Invalid API key")
        raise HTTPException(status_code=403, detail="You are not authorized to access this endpoint")
