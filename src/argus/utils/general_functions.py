from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_logs_dir():
    """Get the logs directory."""
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)  # Create if it doesn't exist
    return logs_dir
