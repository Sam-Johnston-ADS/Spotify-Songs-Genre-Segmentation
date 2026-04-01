# 🎵 Spotify Genre Segmentation & Recommendation System (ML Dashboard)

This project is an end-to-end **Machine Learning + Data Analytics Dashboard** built using **Python, Flask, HTML, CSS, and Scikit-learn**.  
It analyzes Spotify audio features, performs clustering, predicts song genres, and recommends similar songs.

---

## 🚀 Features

✅ Data Cleaning & Preprocessing (ETL Pipeline)  
✅ Exploratory Data Analysis (EDA) + Visualizations  
✅ Correlation Analysis (Heatmap)  
✅ K-Means Clustering (Unsupervised Learning)  
✅ PCA Visualization (2D Cluster Plot)  
✅ Genre Prediction using Random Forest Classifier  
✅ Confusion Matrix + Classification Report  
✅ Multi-class ROC-AUC Curve (Optional)  
✅ Song Recommendation System using KNN  
✅ Web Dashboard using Flask + HTML + CSS  
✅ Search & Filter songs from dataset  
✅ Download filtered songs as CSV  
✅ Interactive colorful UI with animations

---

## 📂 Project Structure

```bash
Spotify-Genre-Segmentation/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│       └── spotify_cleaned.csv
│
├── models/
│   ├── scaler.pkl
│   ├── random_forest_model.pkl
│   ├── label_encoder.pkl
│   ├── knn_recommendation_model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda_visualization.py
│   ├── clustering.py
│   ├── model_training.py
│   ├── recommendation.py
│   ├── save_models.py
│   ├── load_models.py
│   └── utils.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── plots/
│       ├── genre_distribution.png
│       ├── avg_tempo.png
│       ├── scatter_energy_dance.png
│       └── pca_clusters.png
│
└── downloads/
    └── filtered_songs.csv
```

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/Spotify-Genre-Segmentation.git
cd Spotify-Genre-Segmentation
```
### 2️⃣ Install required libraries
```bash
pip install -r requirements.txt
```
## 🧠 Train & Save Models

### Before running the dashboard, generate model files:
```bash
python src/save_models.py
```
This will generate .pkl files inside the models/ folder.

### 🌐 Run Web Application (Flask Dashboard)

Run:
```bash
python app.py
```
Open browser:
```bash
http://127.0.0.1:5000/
```
### 🎯 Machine Learning Models Used
#### 🔹 Random Forest Classifier
Used for predicting playlist genre
Evaluation:
Accuracy
Classification Report
Confusion Matrix
#### 🔹 K-Means Clustering
Used for segmenting songs into clusters
#### 🔹 KNN Recommendation System
Used for finding similar songs based on audio features
### 📈 Visualizations Included

✔ Genre Distribution
✔ Average Tempo by Genre
✔ Danceability vs Energy Scatter Plot
✔ PCA Cluster Visualization

### 🎵 Web Dashboard Output

#### The dashboard allows you to:

Browse songs from dataset
-Search by song name / artist
-Filter songs by genre
-Download filtered songs as CSV
-Predict genre using ML
-Recommend similar songs

###  🛠 Tech Stack
-Python
-Pandas / NumPy
-Matplotlib / Seaborn
-Scikit-learn
-Flask
-HTML / CSS
-Joblib (Model Saving)

### 👨‍💻 Author

Sam Johnston C
B.Tech Artificial Intelligence and Data Science
St. Joseph College of Engineering

### ⭐ Future Improvements
-Add real-time Spotify API integration
-Add song preview and audio player
-Deploy on Heroku / Render
-Add deep learning model for better accuracy
-Add full interactive charts using Plotly

### 📜 License

This project is open-source and free to use for learning and educational purposes.


---
```bash
If you want, I can also generate:
✅ `.gitignore` file  
✅ GitHub project description + tags  
✅ Screenshots section format for README
```
