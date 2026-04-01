import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def recommend_songs(data_path, song_name, n_recommendations=5):
    df = pd.read_csv(data_path)

    features = ["danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness",
                "valence", "tempo"]

    X = df[features]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KNN model
    knn = NearestNeighbors(n_neighbors=n_recommendations + 1, metric="euclidean")
    knn.fit(X_scaled)

    # Check song exists
    if song_name not in df["track_name"].values:
        print("❌ Song not found in dataset!")
        return

    # Index of input song
    idx = df[df["track_name"] == song_name].index[0]

    # Find nearest songs
    distances, indices = knn.kneighbors([X_scaled[idx]])

    print("\n🎵 Input Song:", df.loc[idx, "track_name"])
    print("🎤 Artist:", df.loc[idx, "track_artist"])
    print("🎧 Genre:", df.loc[idx, "playlist_genre"])

    print("\n✅ Recommended Songs:\n")

    for i in range(1, len(indices[0])):  # skipping itself
        rec_index = indices[0][i]
        print(f"{i}. {df.loc[rec_index, 'track_name']} | {df.loc[rec_index, 'track_artist']} | Genre: {df.loc[rec_index, 'playlist_genre']}")


if __name__ == "__main__":
    recommend_songs(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv", "Shape of You")