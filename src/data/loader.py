import os
import pandas as pd

# Define file paths
RAW_DATA_PATH = os.path.join("data", "raw", "movies.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_movies.csv")

def load_raw_data():
    """
    Load the raw dataset from the specified path.
    
    Returns:
        pd.DataFrame: Raw dataset loaded into a DataFrame.
    """
    try:
        data = pd.read_csv(RAW_DATA_PATH)
        print(f"Raw data loaded successfully from {RAW_DATA_PATH}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {RAW_DATA_PATH}")
        return None

def load_processed_data():
    """
    Load the processed dataset from the specified path.
    
    Returns:
        pd.DataFrame: Processed dataset loaded into a DataFrame.
    """
    try:
        data = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Processed data loaded successfully from {PROCESSED_DATA_PATH}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {PROCESSED_DATA_PATH}")
        return None

def save_processed_data(df):
    """
    Save the processed dataset to the specified path.
    
    Args:
        df (pd.DataFrame): The processed DataFrame to save.
    """
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved successfully to {PROCESSED_DATA_PATH}")
