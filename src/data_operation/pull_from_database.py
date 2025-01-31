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
    """
    @abstractmethod
    def pull(self, database_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

class PullDataFromDatabaseQuery(IPullDataFromDatabaseQuery):
    """
    Implementation of `IPullDataFromDatabaseQuery` to pull data from an SQLite database.
    """
    def pull(self, database_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Pulls data from an SQLite database and combines Arabic and English data.
        """
        if not os.path.exists(database_path):
            error_msg = f"The database file does not exist: {database_path}"
            HighLevelErrors.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            DataOperation.info("Connecting to the SQLite database.")
            with sqlite3.connect(database_path) as conn:
                tables = ["Services_Arabic", "Branches_Arabic", "SocialMedia_Arabic", 
                          "Services_English", "Branches_English", "SocialMedia_English"]
                dataframes = {}
                
                for table in tables:
                    try:
                        dataframes[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                        DataOperation.info(f"Data retrieved successfully from {table}.")
                    except Exception:
                        DataOperation.warning(f"Table {table} does not exist or could not be read.")
                        dataframes[table] = pd.DataFrame()
                
                # Combine Arabic and English data if available
                services_df = pd.concat(
                    [dataframes["Services_Arabic"], dataframes["Services_English"]],
                    keys=["Arabic", "English"], names=["Language"]
                ) if not dataframes["Services_Arabic"].empty or not dataframes["Services_English"].empty else pd.DataFrame()
                
                branches_df = pd.concat(
                    [dataframes["Branches_Arabic"], dataframes["Branches_English"]],
                    keys=["Arabic", "English"], names=["Language"]
                ) if not dataframes["Branches_Arabic"].empty or not dataframes["Branches_English"].empty else pd.DataFrame()
                
                social_media_df = pd.concat(
                    [dataframes["SocialMedia_Arabic"], dataframes["SocialMedia_English"]],
                    keys=["Arabic", "English"], names=["Language"]
                ) if not dataframes["SocialMedia_Arabic"].empty or not dataframes["SocialMedia_English"].empty else pd.DataFrame()
                
                DataOperation.info("Data combined successfully.")
                return services_df, branches_df, social_media_df

        except sqlite3.OperationalError as e:
            error_msg = f"Database operation failed: {str(e)}"
            HighLevelErrors.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            HighLevelErrors.error(error_msg)
            raise

def main():
    """
    Main function to execute the data retrieval process.
    """
    database_path = "/workspaces/Chatbot-For-Dental-Clinic/Data/Database/ara_database.sqlite"
    DataOperation.info("Starting the data pull process.")
    try:
        data_puller = PullDataFromDatabaseQuery()
        services_df, branches_df, social_media_df = data_puller.pull(database_path)
        DataOperation.info("Data pulled successfully.")
    except FileNotFoundError as e:
        HighLevelErrors.error(f"File not found: {str(e)}")
        sys.exit(1)
    except Exception as e:
        HighLevelErrors.error(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
