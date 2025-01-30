import os
import sys
import sqlite3
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod

# Define MAIN_DIR to point to the project root directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import DataOperation, HighLevelErrors


class IPullDataFromDatabaseQuery(ABC):
    """
    Interface for pulling data from a database using SQL queries.
    
    This abstract base class enforces the implementation of the `pull` method 
    that retrieves data from a specified database.

    Attributes:
    -----------
    None

    Methods:
    --------
    pull(database_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Retrieves data from an SQLite database located at `database_path`.
        Returns three DataFrames containing services, branches, and social media data, 
        combining both Arabic and English versions of each table.
    """
    
    @abstractmethod
    def pull(self, database_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Abstract method to retrieve data from the database.

        Parameters:
        -----------
        database_path (str): The path to the SQLite database file.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - services_df: Combined Arabic and English services data.
            - branches_df: Combined Arabic and English branches data.
            - social_media_df: Combined Arabic and English social media data.
        """
        pass


class PullDataFromDatabaseQuery(IPullDataFromDatabaseQuery):
    """
    Implementation of `IPullDataFromDatabaseQuery` to pull data from an SQLite database.
    
    This class implements the `pull` method to extract data from specified tables in 
    the SQLite database, handling both Arabic and English versions of the data.
    
    Attributes:
    -----------
    None

    Methods:
    --------
    pull(database_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Pulls data from the database and combines both Arabic and English data into 
        three DataFrames: services, branches, and social media data.
    """
    
    def pull(self, database_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Pulls data from an SQLite database and combines Arabic and English data 
        into DataFrames.

        Parameters:
        -----------
        database_path (str): The path to the SQLite database file.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - services_df: Combined DataFrame containing Arabic and English services data.
            - branches_df: Combined DataFrame containing Arabic and English branches data.
            - social_media_df: Combined DataFrame containing Arabic and English social media data.

        Raises:
        -------
        FileNotFoundError: If the database file does not exist.
        ValueError: If there is an error while executing SQL queries (e.g., invalid table name).
        sqlite3.OperationalError: If there is an error during the database operation.
        """
        
        # Check if the provided database path exists
        if not os.path.exists(database_path):
            error_msg = f"The database file does not exist: {database_path}"
            HighLevelErrors.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            DataOperation.info("Connecting to the SQLite database.")
            conn = sqlite3.connect(database_path)

            try:
                # Read Arabic and English tables into separate DataFrames
                services_arabic_df = pd.read_sql_query("SELECT * FROM Services_Arabic", conn)
                branches_arabic_df = pd.read_sql_query("SELECT * FROM Branches_Arabic", conn)
                socialmedia_arabic_df = pd.read_sql_query("SELECT * FROM SocialMedia_Arabic", conn)
                services_english_df = pd.read_sql_query("SELECT * FROM Services_English", conn)
                branches_english_df = pd.read_sql_query("SELECT * FROM Branches_English", conn)
                socialmedia_english_df = pd.read_sql_query("SELECT * FROM SocialMedia_English", conn)

                DataOperation.info("Data retrieved successfully from all tables.")

                # Combine Arabic and English data into one DataFrame for each category
                services_df = pd.concat(
                    [services_arabic_df, services_english_df],
                    keys=["Arabic", "English"],
                    names=["Language"]
                )
                branches_df = pd.concat(
                    [branches_arabic_df, branches_english_df],
                    keys=["Arabic", "English"],
                    names=["Language"]
                )
                social_media_df = pd.concat(
                    [socialmedia_arabic_df, socialmedia_english_df],
                    keys=["Arabic", "English"],
                    names=["Language"]
                )

                DataOperation.info("Data combined successfully.")
                return services_df, branches_df, social_media_df

            except pd.io.sql.DatabaseError as e:
                # Handle errors related to SQL query execution
                error_msg = "SQL query execution failed. Check table existence."
                HighLevelErrors.error(f"{error_msg} Details: {str(e)}")
                raise ValueError(error_msg) from e

            finally:
                # Close the database connection after reading data
                conn.close()
                DataOperation.info("Database connection closed.")

        except sqlite3.OperationalError as e:
            # Handle errors related to database operations
            error_msg = f"Database operation failed. Details: {str(e)}"
            HighLevelErrors.error(error_msg)
            raise

        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error: {str(e)}"
            HighLevelErrors.error(error_msg)
            raise


def main():
    """
    Main function to execute the data retrieval process.

    This function initializes an instance of the `PullDataFromDatabaseQuery` class, 
    pulls data from the database, and handles any potential exceptions that may occur.
    If the data is successfully retrieved, it logs a success message.
    
    The program exits gracefully in case of any failure.

    Raises:
    -------
    SystemExit: If there is an error during the execution.
    """
    
    database_path = "/workspaces/Chatbot-For-Dental-Clinic/Data/Database/ara_database.sqlite"

    # Start the data pull process
    DataOperation.info("Starting the data pull process.")
    try:
        # Create instance of data puller and retrieve data
        data_puller = PullDataFromDatabaseQuery()
        services_df, branches_df, social_media_df = data_puller.pull(database_path)
        DataOperation.info("Data pulled successfully.")

    except FileNotFoundError as e:
        # Handle file not found error
        HighLevelErrors.error(f"File not found: {str(e)}")
        sys.exit(1)

    except Exception as e:
        # Handle any other exceptions during execution
        HighLevelErrors.error(f"Error during execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    main()
