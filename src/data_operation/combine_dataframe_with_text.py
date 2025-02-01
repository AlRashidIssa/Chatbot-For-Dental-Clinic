import os
import sys
import pandas as pd
from abc import ABC, abstractmethod

# Get the absolute path to the directory two levels above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import DataOperation, HighLevelErrors

class ICombinedTables(ABC):
    """
    Interface for combining specified columns of a DataFrame into a single column.
    """

    @abstractmethod
    def combined(self, columns: list, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to combine specified columns of a DataFrame into a new column.

        Args:
            columns (list): A list of column names to combine.
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The modified DataFrame with a new 'combined' column.
        """
        pass

class CombinedTables(ICombinedTables):
    """
    Class to combine specified columns of a DataFrame into a single column named 'combined'.
    """

    def combined(self, columns: list, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines specified columns of the DataFrame into a single column named 'combined'.

        Args:
            columns (list): A list of column names to combine.
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The modified DataFrame with a new 'combined' column.

        Raises:
            ValueError: If `columns` is not a list, `df` is not a DataFrame,
                        or if any column in `columns` does not exist in `df`.
        """
        DataOperation.info("Starting to combine columns.")

        # Input validation
        if not isinstance(columns, list):
            HighLevelErrors.error("The 'columns' parameter must be a list of column names.")
            raise ValueError("The 'columns' parameter must be a list of column names.")

        if not isinstance(df, pd.DataFrame):
            HighLevelErrors.error("The 'df' parameter must be a pandas DataFrame.")
            raise ValueError("The 'df' parameter must be a pandas DataFrame.")

        if not columns:
            HighLevelErrors.error("The 'columns' list is empty. Provide at least one column name.")
            raise ValueError("The 'columns' list is empty. Provide at least one column name.")

        for col in columns:
            if col not in df.columns:
                HighLevelErrors.error(f"Column '{col}' does not exist in the DataFrame.")
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        # Combine the specified columns into a new column named 'combined'
        df_copy = df.copy()  # Ensure original DataFrame is not modified
        df_copy["combined"] = df_copy[columns].astype(str).agg(' '.join, axis=1)

        DataOperation.info(f"Successfully combined columns: {columns}")
        
        # Save to CSV
        MAIN_DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        data_dir = os.path.join(MAIN_DIR_ROOT, "Data")
        os.makedirs(data_dir, exist_ok=True)  # Ensure directory exists

        csv_path = os.path.join(data_dir, f"data{columns[-1]}.csv")
        df_copy.to_csv(csv_path, index=False)
        DataOperation.info(f"Successfully saved DataFrame to CSV: {csv_path}")

        return df_copy

# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "Los Angeles", "Chicago"]
    }
    df = pd.DataFrame(data)

    # Columns to combine
    columns_to_combine = ["name", "age", "city"]

    # Initialize CombinedTables class
    combiner = CombinedTables()
    result_df = combiner.combined(columns_to_combine, df)

    # Print the result
    print(result_df)
