import os
import sys
# Set the base directory relative to the script's location
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import DataOperation, HighLevelErrors

def get_size(file_path: str) -> str:
    """
    Function to get the size of the file.

    Parameters:
    -----------
    file_path (str): The path of the file for which size is to be calculated.

    Returns:
    --------
    str: The size of the file in MB as a string.

    Raises:
    -------
    FileNotFoundError: If the file does not exist at the given path.
    TypeError: If the file_path is not a string.
    ValueError: If the file size is zero or None.
    Exception: For any other unforeseen errors.
    """
    # Check if file_path is a string
    if not isinstance(file_path, str):
        error_msg = f"The file path must be a string, not: {type(file_path)}"
        HighLevelErrors.error(error_msg)
        raise TypeError(error_msg)

    # Check if the file exists
    if not os.path.exists(file_path):
        error_msg = f"The file at the path '{file_path}' was not found."
        HighLevelErrors.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Get the file size in bytes
        file_size_bytes = os.path.getsize(file_path)

        # Convert file size to MB (from bytes)
        file_size_mb = file_size_bytes / (1024 ** 2)
        
        # If the file size is 0 or None, raise an error
        if file_size_mb <= 0.001 or file_size_bytes is None:
            error_msg = f"File size of '{file_path}' is zero or None."
            HighLevelErrors.error(error_msg)
            raise ValueError(error_msg)

        # Convert file size to GB (from bytes)
        file_size_gb = file_size_bytes / (1024 ** 3)

        # Return the size as a string in GB
        return f"The size of the file is {file_size_mb:.3f} MB."

    except Exception as e:
        # Catch any other exceptions
        error_msg = f"An unexpected error occurred while getting the size of the file '{file_path}': {str(e)}"
        HighLevelErrors.error(error_msg)
        raise Exception(error_msg) from e

if __name__ == "__main__":
    size = get_size("/workspaces/Chatbot-For-Dental-Clinic/Data/Database/ara_database.sqlite")
    print(size)