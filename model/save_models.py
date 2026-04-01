import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def save_all_models(data_path):
    df = pd.read_csv(data_path)

    features = ["danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness",
                "valence", "tempo"]

    X = df[features]

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # --------------------------
    # 1. Save Scaler
    # --------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\scaler.pkl",)
    print("✅ scaler.pkl saved")

    # --------------------------
    # 2. Save KMeans Model
    # --------------------------
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)

    joblib.dump(kmeans, r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\kmeans_model.pkl")
    print("✅ kmeans_model.pkl saved")

    # --------------------------
    # 3. Save KNN Recommendation Model
    # --------------------------
    knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn.fit(X_scaled)

    joblib.dump(knn,r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\knn_recommendation_model.pkl")
    print("✅ knn_recommendation_model.pkl saved")

    # --------------------------
    # 4. Save Random Forest Model
    # --------------------------
    y = df["playlist_genre"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    joblib.dump(rf,r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\random_forest_model.pkl")
    print("✅ random_forest_model.pkl saved")

    # Save label encoder also (important)
    joblib.dump(le,r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\models\label_encoder.pkl")
    print("✅ label_encoder.pkl saved")


if __name__ == "__main__":
    save_all_models(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv")