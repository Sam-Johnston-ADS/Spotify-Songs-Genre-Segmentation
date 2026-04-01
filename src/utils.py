import pandas as pd


def load_data(path):
    """Load CSV file and return dataframe"""
    return pd.read_csv(path)


def save_data(df, path):
    """Save dataframe into CSV file"""
    df.to_csv(path, index=False)
    print("✅ File saved at:", path)


def check_missing_values(df):
    """Print missing values count"""
    print("\nMissing Values:\n", df.isnull().sum())


def check_duplicates(df):
    """Print duplicate rows count"""
    print("\nDuplicate Rows:", df.duplicated().sum())