from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from sklearn.decomposition import PCA

app = Flask(__name__)

# ── Load dataset ────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/spotify_cleaned.csv")

# ── Load ML models ──────────────────────────────────────────────────────────
scaler = joblib.load("models/scaler.pkl")
rf     = joblib.load("models/random_forest_model.pkl")
le     = joblib.load("models/label_encoder.pkl")
knn    = joblib.load("models/knn_recommendation_model.pkl")

features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

os.makedirs("static/plots", exist_ok=True)
os.makedirs("downloads", exist_ok=True)


# ── Plot theme ───────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0a0a0f",
    "surface":  "#111118",
    "surface2": "#18181f",
    "border":   "#1f1f29",
    "text1":    "#f4f4f6",
    "text2":    "#9898a8",
    "text3":    "#5a5a6a",
    "green":    "#1db954",
    "cyan":     "#00d4ff",
    "purple":   "#8b5cf6",
    "pink":     "#f472b6",
    "amber":    "#fbbf24",
    "red":      "#ef4444",
    "orange":   "#fb923c",
}

GENRE_COLORS = {
    "pop":   PALETTE["pink"],
    "rap":   PALETTE["amber"],
    "rock":  PALETTE["red"],
    "latin": PALETTE["orange"],
    "r&b":   PALETTE["purple"],
    "edm":   PALETTE["cyan"],
}

def apply_dark_theme(fig, ax, title="", xlabel="", ylabel=""):
    """Apply consistent dark theme to a figure."""
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface2"])

    ax.set_title(title, color=PALETTE["text1"], fontsize=13, fontweight="bold",
                 fontfamily="sans-serif", pad=14)
    ax.set_xlabel(xlabel, color=PALETTE["text2"], fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=PALETTE["text2"], fontsize=10, labelpad=8)

    ax.tick_params(colors=PALETTE["text3"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    ax.grid(True, color=PALETTE["border"], linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def generate_plots():

    genres = df["playlist_genre"].value_counts()
    genre_names = [g.title() for g in genres.index]
    colors = [GENRE_COLORS.get(g, PALETTE["green"]) for g in genres.index]

    # ── 1. Genre Distribution ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(genre_names, genres.values, color=colors,
                  edgecolor="none", width=0.6, zorder=3)

    for bar, val in zip(bars, genres.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 120,
                f"{val:,}", ha="center", va="bottom",
                color=PALETTE["text2"], fontsize=9)

    apply_dark_theme(fig, ax, "Genre Distribution", "Genre", "Track Count")
    ax.set_ylim(0, genres.values.max() * 1.18)
    plt.tight_layout()
    plt.savefig("static/plots/genre_distribution.png", dpi=130,
                bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close()

    # ── 2. Average Tempo by Genre ───────────────────────────────────────────
    avg_tempo = df.groupby("playlist_genre")["tempo"].mean().sort_values()
    bar_colors = [GENRE_COLORS.get(g, PALETTE["green"]) for g in avg_tempo.index]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh([g.title() for g in avg_tempo.index], avg_tempo.values,
                   color=bar_colors, edgecolor="none", height=0.55, zorder=3)

    for bar, val in zip(bars, avg_tempo.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", color=PALETTE["text2"], fontsize=9)

    apply_dark_theme(fig, ax, "Average Tempo by Genre", "Tempo (BPM)", "")
    ax.set_xlim(0, avg_tempo.max() * 1.15)
    plt.tight_layout()
    plt.savefig("static/plots/avg_tempo.png", dpi=130,
                bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close()

    # ── 3. Danceability vs Energy Scatter ───────────────────────────────────
    sample = df.sample(min(3000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    for genre, grp in sample.groupby("playlist_genre"):
        ax.scatter(grp["danceability"], grp["energy"],
                   label=genre.title(),
                   color=GENRE_COLORS.get(genre, PALETTE["green"]),
                   alpha=0.45, s=14, edgecolors="none", zorder=3)

    apply_dark_theme(fig, ax, "Danceability vs Energy", "Danceability", "Energy")

    legend = ax.legend(
        framealpha=0, labelcolor=PALETTE["text2"],
        fontsize=9, loc="lower right",
        markerscale=1.4
    )
    plt.tight_layout()
    plt.savefig("static/plots/scatter_energy_dance.png", dpi=130,
                bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close()

    # ── 4. PCA Cluster Visualization ────────────────────────────────────────
    X = df[features]
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]

    sample_pca = df.sample(min(4000, len(df)), random_state=99)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for genre, grp in sample_pca.groupby("playlist_genre"):
        ax.scatter(grp["pca_1"], grp["pca_2"],
                   label=genre.title(),
                   color=GENRE_COLORS.get(genre, PALETTE["green"]),
                   alpha=0.4, s=12, edgecolors="none", zorder=3)

    var_exp = pca.explained_variance_ratio_
    apply_dark_theme(
        fig, ax,
        "PCA Cluster Visualization",
        f"PC 1  ({var_exp[0]*100:.1f}% variance)",
        f"PC 2  ({var_exp[1]*100:.1f}% variance)"
    )
    legend = ax.legend(
        framealpha=0, labelcolor=PALETTE["text2"],
        fontsize=9, loc="upper right"
    )
    plt.tight_layout()
    plt.savefig("static/plots/pca_clusters.png", dpi=130,
                bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close()

    # Drop temp PCA columns so they don't interfere
    df.drop(columns=["pca_1", "pca_2"], inplace=True, errors="ignore")


generate_plots()


# ── Recommendation ──────────────────────────────────────────────────────────
def recommend_songs(song_name):
    matches = df[df["track_name"].str.lower() == song_name.strip().lower()]
    if matches.empty:
        return []

    X = df[features]
    X_scaled = scaler.transform(X)

    idx = matches.index[0]
    distances, indices = knn.kneighbors([X_scaled[idx]])

    rec_songs = []
    for i in range(1, len(indices[0])):
        rec_index = indices[0][i]
        rec_songs.append({
            "track_name":    df.loc[rec_index, "track_name"],
            "track_artist":  df.loc[rec_index, "track_artist"],
            "playlist_genre": df.loc[rec_index, "playlist_genre"]
        })
    return rec_songs


# ── Home Route ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    predicted_genre = None
    error_message   = None
    recommended_list = []
    rec_error        = None

    search_query   = request.args.get("search", "").lower()
    selected_genre = request.args.get("genre", "")

    filtered_df = df.copy()

    if search_query:
        filtered_df = filtered_df[
            filtered_df["track_name"].str.lower().str.contains(search_query, na=False) |
            filtered_df["track_artist"].str.lower().str.contains(search_query, na=False)
        ]
    if selected_genre:
        filtered_df = filtered_df[filtered_df["playlist_genre"] == selected_genre]

    songs      = filtered_df[["track_name", "track_artist", "playlist_genre",
                               "tempo", "danceability"]].head(50)
    genre_list = sorted(df["playlist_genre"].unique())

    if request.method == "POST":
        if "danceability" in request.form:
            try:
                values = [float(request.form[f]) for f in features]
                values_scaled  = scaler.transform([values])
                pred           = rf.predict(values_scaled)
                predicted_genre = le.inverse_transform(pred)[0]
            except Exception:
                error_message = "Please enter valid numeric values for all fields."

        if "song_name" in request.form:
            song_name        = request.form["song_name"]
            recommended_list = recommend_songs(song_name)
            if not recommended_list:
                rec_error = f'Song "{song_name}" not found in the dataset.'

    return render_template(
        "index.html",
        songs           = songs.to_dict(orient="records"),
        genre_list      = genre_list,
        predicted_genre = predicted_genre,
        error_message   = error_message,
        selected_genre  = selected_genre,
        search_query    = search_query,
        recommended_list= recommended_list,
        rec_error       = rec_error
    )


# ── Download Route ──────────────────────────────────────────────────────────
@app.route("/download")
def download_csv():
    search_query   = request.args.get("search", "").lower()
    selected_genre = request.args.get("genre", "")

    filtered_df = df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df["track_name"].str.lower().str.contains(search_query, na=False) |
            filtered_df["track_artist"].str.lower().str.contains(search_query, na=False)
        ]
    if selected_genre:
        filtered_df = filtered_df[filtered_df["playlist_genre"] == selected_genre]

    file_path = "downloads/filtered_songs.csv"
    filtered_df.to_csv(file_path, index=False)
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
