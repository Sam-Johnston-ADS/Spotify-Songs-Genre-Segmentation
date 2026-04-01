import pandas as pd
import os

def preprocess_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)
    print("Before Cleaning Shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("After Cleaning Shape:", df.shape)

    # Create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print("✅ Cleaned dataset saved at:", output_path)

    return df


if __name__ == "__main__":
    preprocess_data(
        r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\raw\spotify dataset.csv",
        r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv"
    )