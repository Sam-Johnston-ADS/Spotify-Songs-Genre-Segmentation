from src.data_preprocessing import preprocess_data
from src.eda_visualization import eda_visualization
from src.clustering import kmeans_clustering
from src.model_training import train_model
from src.recommendation import recommend_songs


def main():
    print("========== SPOTIFY GENRE SEGMENTATION PROJECT ==========\n")

    # Step 1: Preprocessing
    print("Step 1: Data Preprocessing...\n")
    preprocess_data(
        r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\raw\spotify dataset.csv",
        r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv"
    )

    # Step 2: EDA Visualization
    print("\nStep 2: EDA Visualization...\n")
    eda_visualization(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv")

    # Step 3: Clustering
    print("\nStep 3: KMeans Clustering...\n")
    kmeans_clustering(
         r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv",
        r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\output\spotify_clustered.csv",
        k=5
    )

    # Step 4: Model Training
    print("\nStep 4: Genre Classification Model Training...\n")
    train_model(r"C:\Users\Sam\Desktop\Spotify_Genre_Segmentation\data\processed\spotify_cleaned.csv")

    # Step 5: Recommendation System
    print("\nStep 5: Recommendation System...\n")
    song_name = input("Enter a song name from dataset: ")
    recommend_songs("data/processed/spotify_cleaned.csv", song_name)

    print("\n========== PROJECT COMPLETED ==========")


if __name__ == "__main__":
    main()