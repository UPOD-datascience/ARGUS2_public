import logging
import os


class PathManager_RES_Folder_UPOD:
    """Class to manage file paths in the RES-Folder-UPOD structure across different
    operating systems.

    File paths follow this structure:
    - lab_research\\RES-Folder-UPOD\\{folder}\\E_ResearchData\\2_ResearchData...
    \\{date_folder}\\{subpath}\\{filename}

    Where {subpath} is optional.

    """

    def __init__(self, logger=None):
        if logger is None:
            # Create a default logger if none is provided
            self.logger = logging.getLogger(__name__)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger

        # Base paths for different OS
        if os.name == "nt":  # Windows
            self.base_path = r"L:\lab_research\RES-Folder-UPOD"
        elif os.name == "posix":  # macOS
            self.base_path = "/Volumes/DATA/LAB/lab_research/RES-Folder-UPOD"
        else:
            error_msg = f"Unsupported operating system: {os.name}"
            self.logger.error(error_msg)
            raise OSError(error_msg)

    def get_file_path(self, folder, date_folder, *args):
        """Generate the correct file path based on the operating system.

        Parameters
        ----------
        folder: Folder name of your project in the RES-Folder-UPOD
        date_folder: Date folder within the E_ResearchData/2_ResearchData directory
        *args: Variable arguments - either:
               1. Just filename (if no subpath)
               2. subpath, filename (if subpath exists)

        Returns
        -------
        str: Complete path to the file

        """
        try:
            # Build the base path
            path_elements = [
                self.base_path,
                folder,
                "E_ResearchData",
                "2_ResearchData",
                date_folder,
            ]

            # Handle variable arguments
            if len(args) == 1:
                # Only filename was provided
                filename = args[0]
            elif len(args) == 2:
                # Both subpath and filename were provided
                subpath, filename = args

                # Process subpath if it exists
                if subpath and subpath.strip():
                    if isinstance(subpath, list):
                        path_elements.extend(subpath)
                    elif isinstance(subpath, str):
                        subpath_elements = subpath.replace("\\", "/").split("/")
                        subpath_elements = [e for e in subpath_elements if e.strip()]
                        path_elements.extend(subpath_elements)
            else:
                error_msg = f"Invalid number of arguments: expected 3 or 4, got {len(args) + 2}"
                self.logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                return None

            # Add filename
            path_elements.append(filename)

            # Join all path elements
            full_path = os.path.join(*path_elements)

            # Check if the folder exists
            folder_path = os.path.dirname(full_path)
            if not os.path.exists(folder_path):
                error_msg = f"Folder does not exist: {folder_path}"
                self.logger.error(error_msg)
                # print(f"ERROR: {error_msg}")

            # Check if the file exists
            if not os.path.exists(full_path):
                error_msg = f"File does not exist: {full_path}"
                self.logger.error(error_msg)
                # print(f"ERROR: {error_msg}")

            return full_path

        except Exception as e:
            error_msg = f"Error generating file path: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            print(f"Cause: {e.__class__.__name__}")
            return None
