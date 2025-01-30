import os, sys, zipfile
from pathlib import Path
from abc import ABC, abstractmethod

# Define MAIN_DIR to point to the project root directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import HighLevelErrors, DataOperation


class IUnzipFile(ABC):
    """
    Abstract Base Class (ABC) for unzipping files.
    """

    @abstractmethod
    def unzip(self, zip_path: str, extract_to: str) -> None:
        """
        Abstract method to unzip a file.

        Parameters:
        -----------
        zip_path (str): The path to the zip file.
        extract_to (str): The directory where the contents should be extracted.

        Returns:
        --------
        None
        """
        pass


class UnzipFile(IUnzipFile):
    """
    Concrete implementation of IUnzipFile for unzipping files.
    """

    def unzip(self, zip_path: str, extract_to: str) -> None:
        """
        Unzips a file to the specified directory.

        Parameters:
        -----------
        zip_path (str): The path to the zip file to be extracted.
        extract_to (str): The directory where the contents should be extracted.

        Raises:
        -------
        FileNotFoundError: If the zip file does not exist.
        NotADirectoryError: If the extract_to path is not a directory.
        zipfile.BadZipFile: If the file is not a valid zip file.
        Exception: For any other unforeseen errors during extraction.

        Returns:
        --------
        None
        """
        # Validate parameters
        if not isinstance(zip_path, str):
            error_msg = f"The zip file path must be a string. Provided type: {type(zip_path)}"
            HighLevelErrors.error(error_msg)
            raise TypeError(error_msg)

        if not isinstance(extract_to, str):
            error_msg = f"The extraction path must be a string. Provided type: {type(extract_to)}"
            HighLevelErrors.error(error_msg)
            raise TypeError(error_msg)

        # Check if the zip file exists
        if not os.path.exists(zip_path):
            error_msg = f"The specified zip file does not exist: {zip_path}"
            HighLevelErrors.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Check if the extraction directory exists
        if not os.path.isdir(extract_to):
            error_msg = f"The extraction path is not a valid directory: {extract_to}"
            HighLevelErrors.error(error_msg)
            raise NotADirectoryError(error_msg)

        try:
            # Attempt to open and extract the zip file
            DataOperation.info("Unzipping file started ...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:                
                # Extract all contents to the specified directory
                zip_ref.extractall(path=extract_to)
                DataOperation.info("Unzip file complated ...")


        except zipfile.BadZipFile:
            error_msg = f"The file is not a valid zip file or is corrupted: {zip_path}"
            HighLevelErrors.error(error_msg)
            raise zipfile.BadZipFile(error_msg)

        except Exception as e:
            error_msg = f"An unexpected error occurred during extraction: {str(e)}"
            HighLevelErrors.error(error_msg)
            raise Exception(error_msg) from e


if __name__ == "__main__":
    zip_path="/workspaces/Chatbot-For-Dental-Clinic/Data/database_v1.zip"
    extract_to="/workspaces/Chatbot-For-Dental-Clinic/Data"

    DataOperation.info(f"Starting extraction of zip file: {zip_path}")
    unzipping = UnzipFile().unzip(zip_path=zip_path,
                                  extract_to=extract_to)
    DataOperation.info(f"Extraction completed successfully. Contents extracted to: {extract_to}")