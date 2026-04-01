import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_visualization(data_path):
    df = pd.read_csv(data_path)

    print("Dataset Shape:", df.shape)
    print(df.head())

    # 1. Genre Distribution
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, x="playlist_genre")
    plt.xticks(rotation=45)
    plt.title("Genre Distribution")
    plt.show()

    # 2. Histogram of Numeric Features
    df.hist(figsize=(18,12), bins=20)
    plt.suptitle("Histogram of Numeric Features")
    plt.show()

    # 3. Boxplot for Danceability
    plt.figure(figsize=(10,5))
    sns.boxplot(x=df["danceability"])
    plt.title("Boxplot of Danceability")
    plt.show()

    # 4. Scatter Plot (Danceability vs Energy)
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x="danceability", y="energy", hue="playlist_genre")
    plt.title("Danceability vs Energy by Genre")
    plt.show()


if __name__ == "__main__":
    eda_visualization(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv")