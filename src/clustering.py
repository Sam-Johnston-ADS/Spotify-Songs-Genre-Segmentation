import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def kmeans_clustering(data_path, output_path, k=5):
    df = pd.read_csv(data_path)

    features = ["danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness",
                "valence", "tempo"]

    X = df[features]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans Model
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Cluster distribution
    print("Cluster Distribution:\n")
    print(df["cluster"].value_counts())

    # Create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save clustered dataset
    df.to_csv(output_path, index=False)
    print("\n✅ Clustered dataset saved at:", output_path)

    return df


if __name__ == "__main__":
    kmeans_clustering(
        r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv",
        r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\output\spotify_clustered.csv",
        k=5
    )