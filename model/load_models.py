import pandas as pd
import joblib
from sklearn.metrics.pairwise import euclidean_distances


# Load saved models
scaler = joblib.load(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\scaler.pkl")
kmeans = joblib.load(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\kmeans_model.pkl")
knn = joblib.load(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\knn_recommendation_model.pkl")
rf = joblib.load(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\random_forest_model.pkl")
le = joblib.load(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\label_encoder.pkl")


# Features used
features = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness",
            "valence", "tempo"]


def predict_genre(song_features):
    """
    song_features = list of 9 values in same order as features list
    """
    song_scaled = scaler.transform([song_features])
    pred = rf.predict(song_scaled)
    genre = le.inverse_transform(pred)[0]
    return genre


def predict_cluster(song_features):
    song_scaled = scaler.transform([song_features])
    cluster = kmeans.predict(song_scaled)[0]
    return cluster


def recommend_songs(song_name, data_path=r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv"):
    df = pd.read_csv(data_path)

    X = df[features]
    X_scaled = scaler.transform(X)

    if song_name not in df["track_name"].values:
        print("❌ Song not found!")
        return

    idx = df[df["track_name"] == song_name].index[0]

    distances, indices = knn.kneighbors([X_scaled[idx]])

    print("\n🎵 Input Song:", df.loc[idx, "track_name"])
    print("🎤 Artist:", df.loc[idx, "track_artist"])
    print("🎧 Genre:", df.loc[idx, "playlist_genre"])

    print("\n✅ Recommended Songs:\n")

    for i in range(1, len(indices[0])):  # skip itself
        rec_index = indices[0][i]
        print(f"{i}. {df.loc[rec_index, 'track_name']} | {df.loc[rec_index, 'track_artist']} | Genre: {df.loc[rec_index, 'playlist_genre']}")


if __name__ == "__main__":
    # Example song features input (dummy example)
    example_song = [0.8, 0.7, -5.0, 0.05, 0.2, 0.0, 0.1, 0.6, 120]

    print("Predicted Genre:", predict_genre(example_song))
    print("Predicted Cluster:", predict_cluster(example_song))

    recommend_songs("lay all your ;ove on me")