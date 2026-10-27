import os

import yaml

from argus.utils.general_functions import get_project_root


def load_config(config_type=None, config_path=None, version=None, mode=None):
    """Load configuration from a YAML file."""
    if config_path is None:
        if config_type is None:
            config_type = "general"
        config_path = os.path.join(get_project_root(), "configs", f"config_{config_type}.yaml")

    try:
        with open(config_path, "r") as file:
            full_config = yaml.safe_load(file)

        # Extract version-specific config if version is specified
        if version and "versions" in full_config:
            if version not in full_config["versions"]:
                raise ValueError(f"Version '{version}' not found in {config_type} configuration")

            version_config = full_config["versions"][version]

            # Create config with common parts (everything except "versions")
            result_config = {k: v for k, v in full_config.items() if k != "versions"}

            # Extract mode-specific config if mode is specified
            if mode and "modes" in version_config:
                if mode not in version_config["modes"]:
                    raise ValueError(f"Mode '{mode}' not found for version '{version}' in {config_type} configuration")

                # Add version-level config (everything except "modes")
                result_config.update({k: v for k, v in version_config.items() if k != "modes"})

                # Add mode-specific config
                result_config.update(version_config["modes"][mode])
            else:
                # No mode specified, add all version config
                result_config.update(version_config)

            return result_config

        return full_config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from None
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}") from e
