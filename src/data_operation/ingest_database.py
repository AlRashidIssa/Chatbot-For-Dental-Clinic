import gdown
import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod

# Define MAIN_DIR to point to the project root directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import DataOperation, HighLevelErrors  # Loges
from utils.get_size import get_size


class ILoadFromDrive(ABC):
    """
    Abstract Base Class (ABC) for loading files from Google Drive.
    """

    @abstractmethod
    def load(self, url: str, save_archive: str, name: str) -> None:
        """
        Abstract method to be implemented for loading files from Google Drive.

        Parameters:
        -----------
        url (str): The Google Drive link for the file.
        save_archive (str): The directory path to save the downloaded file.
        name (str): The name to save the downloaded file as.

        Returns:
        --------
        None
        """
        pass


class LoadFromDrive(ILoadFromDrive):
    """
    Concrete class for downloading files from Google Drive.
    """

    def load(self, url: str, save_archive: str, name: str) -> None:
        """
        Downloads a file from Google Drive and saves it to the specified directory.

        Parameters:
        -----------
        url (str): The Google Drive link for the file to be downloaded.
        save_archive (str): The path to the directory where the file should be saved.
        name (str): The name to use for the saved file (without extension).

        Raises:
        -------
        TypeError: If any parameter is not a string.
        FileNotFoundError: If the save directory does not exist.
        ValueError: If the extracted file ID is invalid or missing.
        Exception: For any other unforeseen errors during the download process.

        Returns:
        --------
        None
        """
        # Validate parameters
        if not isinstance(url, str):
            error_msg = f"The URL must be a string. Provided type: {type(url)}"
            HighLevelErrors.error(error_msg)
            raise TypeError(error_msg)

        if not isinstance(name, str):
            error_msg = f"The file name must be a string. Provided type: {type(name)}"
            HighLevelErrors.error(error_msg)
            raise TypeError(error_msg)
        os.makedirs(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))}/Data")
        os.makedirs(save_archive, exist_ok=True)

        try:
            # Extract the file ID from the Google Drive URL
            try:
                file_id = url.split('/d/')[1].split('/')[0]
                download_link = f"https://drive.google.com/uc?id={file_id}"
            except IndexError:
                error_msg = f"Invalid Google Drive URL format: {url}"
                HighLevelErrors.error(error_msg)
                raise ValueError(error_msg)

            # Prepare the full save path
            save_path = Path(save_archive) / f"{name}.sqlite"

            # Log the download process
            DataOperation.info(f"Starting download from Google Drive. File: {name}, Destination: {save_path}")

            # Download the file using gdown
            gdown.download(download_link, str(save_path), quiet=False)

            # Log successful download
            DataOperation.info(f"File downloaded successfully: {save_path}")

        except Exception as e:
            error_msg = f"An error occurred while downloading the file: {str(e)}"
            HighLevelErrors.error(error_msg)
            raise Exception(error_msg) from e


def main():
    """
    Main entry point for downloading a database file from Google Drive.

    This function logs the process, initializes the downloader, and reports
    the final status along with the file size.
    """
    save_archive = "/workspaces/Chatbot-For-Dental-Clinic/Data"
    DataOperation.info("Starting Database Download from Google Drive")

    try:
        # Create an instance of the loader
        loader = LoadFromDrive()
        loader.load(
            url="https://drive.googlej.com/file/d/139OEjxiFwxJtFQWaixF0fG4JSQyGp6EP/view?usp=sharing",
            save_archive=save_archive,
            name="database_v1"
        )

        # Log final status with file size
        file_path = f"{save_archive}/database_v1.zip"
        DataOperation.info(f"Completed Database Download. "
                      f"Note: Size = {get_size(file_path)}, "
                      f"Location: {save_archive}")
    except Exception as e:
        HighLevelErrors.error(f"Failed to download database. Error: {str(e)}")

if __name__ == "__main__":
    main()
