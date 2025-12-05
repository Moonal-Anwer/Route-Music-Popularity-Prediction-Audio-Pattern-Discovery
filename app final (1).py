import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import joblib
import librosa
import tempfile
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import scipy.sparse as sp
import hickle as hkl

# ============================
# 1. PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Everything you need to know about your music",
    layout="wide",
)

st.title(" Real-Time Song Popularity Prediction")
st.markdown("Upload an MP3 to analyze its features and predict its popularity score.")


# ============================
# 2. LOAD MODELS
# ============================

@st.cache_resource
def load_models():

        xgb_model = joblib.load("xgboost_best.pkl")
        mlp_scaler=joblib.load("scaler_train.joblib")
        scaler = joblib.load("scaler_data.pkl")
        genre_ohe = joblib.load("genre_ohe.pkl")
        scaler_xgb=joblib.load("scaler (1).pkl")
        artist_freq_map = joblib.load("artist_freq.pkl")
        from tensorflow.keras.models import load_model
        mlp_model = load_model("best_mlp_model (1).h5")
        return xgb_model,mlp_scaler, scaler, genre_ohe, scaler_xgb,artist_freq_map, mlp_model



xgb_model, mlp_scaler,scaler, genre_ohe, scaler_xgb,artist_freq_map,mlp_model = load_models()



# getting the exact feature order that the scaler expect
FEATURE_LIST = list(scaler.feature_names_in_)
# getting all 100+ genre categories safely from the OHE object
GENRE_LIST = genre_ohe.categories_[0]


# ============================
# 3. AUDIO EXTRACTION FUNCTION
# ============================

def extract_audio_features(file_path):

        total_duration = librosa.get_duration(path=file_path) #extract the file whole duration


        segment_length = 120  # analyze 120 seconds

        if total_duration > segment_length:
            offset = (total_duration / 2) - (segment_length / 2) # to start from the middle
        else: # handel if audio is shorter than the segmant
            offset = 0
            segment_length = None


        y, sr = librosa.load(file_path, offset=offset, duration=segment_length)


        features = {
            "duration_ms": total_duration * 1000,

            "danceability": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean() / 10000),
            "energy": float(np.mean(y ** 2)),
            "loudness": float(librosa.amplitude_to_db(np.abs(y)).mean()),
            "speechiness": float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean() / 100000),
            "acousticness": float(np.mean(librosa.feature.zero_crossing_rate(y))),
            "instrumentalness": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) / 10000),
            "liveness": float(np.mean(librosa.feature.rms(y=y))),
            "valence": float(np.mean(librosa.feature.spectral_flatness(y=y))),
            "tempo": float(librosa.beat.beat_track(y=y, sr=sr)[0])
        }
        return features

# ============================
# 4. USER INTERFACE

col_input, col_pred = st.columns([1, 1.2])

with col_input:
    st.subheader("1. Song Input")


    uploaded_file = st.file_uploader("Upload MP3 File", type=["mp3", "wav", "m4a"])

    defaults = {
        "danceability": 0.5, "energy": 0.6, "loudness": -8.0,
        "speechiness": 0.05, "acousticness": 0.1, "instrumentalness": 0.0,
        "liveness": 0.1, "valence": 0.5, "tempo": 120.0, "duration_ms": 210000.0
    }

    if uploaded_file:
        with st.spinner("Analyzing the track..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            extracted = extract_audio_features(tmp_path)
            os.remove(tmp_path)

            if extracted:
                defaults.update(extracted)
                st.success(
                    f" Extracted features (Total Duration: {extracted['duration_ms'] / 1000:.1f}s)")

#-----------------------------
#user input

    with st.form("main_form"):
        st.write("### Metadata")
        c1, c2 = st.columns(2)

        all_artists = sorted(artist_freq_map.keys())
        artist_name = st.selectbox("Select Artist", all_artists)
        artist_freq = artist_freq_map.get(artist_name, 0)
        genre = c2.selectbox("Genre", GENRE_LIST)
        st.write("### Audio Features")
        st.caption("Values auto-filled from MP3. change it if needed.")

        dance = st.slider("Danceability", 0.0, 1.0, float(defaults["danceability"]))
        energy = st.slider("Energy", 0.0, 1.0, float(defaults["energy"]))
        acoustic = st.slider("Acousticness", 0.0, 1.0, float(defaults["acousticness"]))
        val = st.slider("Valence (Mood)", 0.0, 1.0, float(defaults["valence"]))

        with st.expander("Advanced Features"):
            tempo = st.number_input("Tempo (BPM)", 0.0, 250.0, float(defaults["tempo"]))
            loudness = st.number_input("Loudness (dB)", -60.0, 0.0, float(defaults["loudness"]))
            speech = st.number_input("Speechiness", 0.0, 1.0, float(defaults["speechiness"]))
            instru = st.number_input("Instrumentalness", 0.0, 1.0, float(defaults["instrumentalness"]))
            liveness = st.number_input("Liveness", 0.0, 1.0, float(defaults["liveness"]))
            explicit = st.checkbox("Explicit Content?", value=False)
            key = st.selectbox("Key", range(12), index=0)
            mode = st.radio("Mode", [0, 1], index=1, help="1=Major, 0=Minor")
            time_sig = st.selectbox("Time Signature", [3, 4, 5, 7], index=1)
            duration_minutes = st.number_input("Duration (minutes)", 0.0, 60.0, float(defaults["duration_ms"]) / 60000)


            duration = duration_minutes * 60 * 1000  # ms

        submit_btn = st.form_submit_button("Predicting Popularity")

# ============================
# 5. PREDICTION LOGIC

with col_pred:
    if submit_btn:
        st.subheader(" Prediction Analysis")


        input_data = {
            "duration_ms": duration,
            "danceability": dance,
            "energy": energy,
            "key": key,
            "loudness": loudness,
            "mode": mode,
            "speechiness": speech,
            "acousticness": acoustic,
            "instrumentalness": instru,
            "liveness": liveness,
            "valence": val,
            "tempo": tempo,
            "time_signature": time_sig,
            "explicit": int(explicit)
        }
        df = pd.DataFrame([input_data])

        df["duration_log"] = np.log1p(df["duration_ms"])
        df["energy_dance_ratio"] = df["energy"] / (df["danceability"] + 1e-5)
        df["acoustic_electronic"] = df["acousticness"] - df["instrumentalness"]


        # if artist not in list defaults to 0
        freq_val = artist_freq_map.get(artist_name, 0)
        df["artist_freq"] = freq_val


        # resting all genre columns to 0
        for cat in GENRE_LIST:
            df[f"track_genre_{cat}"] = 0

        #  selected genre = 1
        selected_genre_col = f"track_genre_{genre}"
        if selected_genre_col in df.columns:
            df[selected_genre_col] = 1

        df_final = df.reindex(columns=FEATURE_LIST, fill_value=0)

        # scaling the data
        df_scaled = scaler.transform(df_final)
        df_scaled_xgb=scaler_xgb.transform(df_scaled)
        df_mlp_input=mlp_scaler.transform(df_scaled)

        # 6. Prediction
        pred_xgb = xgb_model.predict(df_scaled_xgb)[0]
        pred_mlp = mlp_model.predict(df_mlp_input)[0][0]  # Keras returns 2D array

        # Calculate mean prediction
        prediction = (pred_xgb + pred_mlp)
        # 7. Display Results
        st.divider()

        # Big Metric

        st.metric(label="Average Prediction", value=f"{prediction:.1f} / 100")

        # Visual Breakdown
        st.write("---")
        st.caption("Feature Profile:")

        chart_data = pd.DataFrame({
            "Metric": ["Energy", "Danceability", "Acousticness", "Valence", "Instrum."],
            "Value": [energy, dance, acoustic, val, instru]
        })
        st.bar_chart(chart_data, x="Metric", y="Value")

#-----------------------------#


st.set_page_config(page_title="Music Cluster Explorer", layout="wide")


# ---------------------------------
# 1. LOAD EVERYTHING
# ---------------------------------
@st.cache_resource
def load_assets():
    # Load Models
    kmeans = hkl.load("kmeans_model.hkl")
    pca = hkl.load("pca_model.hkl")

    # Load Data
    df = pd.read_csv("Preprocessed_Data.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Load Preprocessors
    scaler = joblib.load("scaler_kkk.pkl")
    cat_vectorizers = joblib.load("cat_vectorizers.pkl")
    feature_names = joblib.load("feature_names.pkl")

    return kmeans, pca, df, scaler, cat_vectorizers, feature_names


kmeans, pca, df, scaler, cat_vectorizers, feature_names = load_assets()


# ---------------------------------
# 2. CREATE SPARSE MATRIX (The Memory Fix)
# ---------------------------------
@st.cache_resource
def create_sparse_matrix(_df, _scaler, _cat_vectorizers, _feature_names):
    """
    Creates a memory-efficient Sparse Matrix instead of a giant DataFrame.
    """

    num_cols = _scaler.feature_names_in_
    X_num = _scaler.transform(_df[num_cols])

    sparse_blocks = [sp.csr_matrix(X_num)]  # start with numericals

    for col, vec in _cat_vectorizers.items():
        col_data = _df[col].astype(str).values
        X_cat = vec.transform(col_data)
        sparse_blocks.append(X_cat)

    X_final = sp.hstack(sparse_blocks)

    return X_final.tocsr()


# Create the matrix
X_sparse = create_sparse_matrix(df, scaler, cat_vectorizers, feature_names)

# ---------------------------------
# 3. PRE-CALCULATE CLUSTERS & PCA
# ---------------------------------
if "cluster" not in df.columns:
    # Predict clusters on the sparse matrix (Fast)
    df["cluster"] = kmeans.predict(X_sparse)

if "pca_x" not in df.columns:
    pca_x_list, pca_y_list = [], []
    batch_size = 100

    # Process in chunks of 100 rows
    for i in range(0, X_sparse.shape[0], batch_size):
        X_batch = X_sparse[i:i + batch_size].toarray()
        coords = pca.transform(X_batch)
        pca_x_list.extend(coords[:, 0])
        pca_y_list.extend(coords[:, 1])

    df["pca_x"] = pca_x_list
    df["pca_y"] = pca_y_list

# ---------------------------------
# 4. APP INTERFACE
# ---------------------------------
st.sidebar.title("Music Cluster ")

# Track Selection
selected_track_name = st.sidebar.selectbox("Choose a track:", df["track_name"].unique())

track_idx = df[df["track_name"] == selected_track_name].index[0]
track_cluster = df.loc[track_idx, "cluster"]

st.header(f"Track: {selected_track_name}")
st.subheader(f"Cluster Assigned → **Cluster {track_cluster}**")

# ---------------------------------
# 5. VISUALIZATION
# ---------------------------------
st.subheader("Cluster Visualization")

fig = px.scatter(
    df,
    x="pca_x",
    y="pca_y",
    color=df["cluster"].astype(str),
    hover_data=["track_name", "artists", "popularity"],
    title="Music Cluster PCA Visualization",
    opacity=0.6
)

# Highlight selected track (Big Red Star)
fig.add_scatter(
    x=[df.loc[track_idx, "pca_x"]],
    y=[df.loc[track_idx, "pca_y"]],
    mode="markers",
    marker=dict(size=25, symbol="star", color="red"),
    name="Selected Track"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# 6. SIMILARITY RECOMMENDATIONS
# ---------------------------------
st.subheader("Similar Tracks ")

selected_vector = X_sparse[track_idx]

cluster_indices = df[df["cluster"] == track_cluster].index
cluster_matrix = X_sparse[cluster_indices]

sim_scores = cosine_similarity(selected_vector, cluster_matrix)[0]

results_df = df.loc[cluster_indices].copy()
results_df["similarity"] = sim_scores

# Show top 10
top_recs = (
    results_df[results_df["track_name"] != selected_track_name]
    .sort_values("similarity", ascending=False)
    .head(10)[["track_name", "artists", "popularity", "similarity"]]
)

st.table(top_recs)

#-----------------------------#
st.set_page_config(page_title="Music Explorer", layout="wide")

# -------------------------------------------------------
# LOAD BOTH DATASETS

@st.cache_data
def load_small_data():
    return pd.read_csv("Preprocessed_Data.csv")

@st.cache_data
def load_large_data():
    return pd.read_csv("train_data_.csv")


df_small = load_small_data()     # has real "artists"
df_large = load_large_data()     # no artists, only features


# -------------------------------------------------------
# SECTION TITLE

st.title(" Music Popularity & Audio Pattern Discovery")
st.write("Artist similarity • Trend spotting • Playlist generator")


# =======================================================
# 1) ARTIST SIMILARITY ANALYSIS
# =======================================================

st.header(" Artist Similarity Analysis")

numeric_features = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo','energy_dance_ratio',
    'acoustic_electronic','log_duration_ms'
]


df_small = df_small.dropna(subset=["artists"])


artist_means = df_small.groupby("artists")[numeric_features].mean()

# scale
scaler = StandardScaler()
scaled = scaler.fit_transform(artist_means)

artists_list = artist_means.index.tolist()

selected_artist = st.selectbox("Pick an Artist", artists_list)

#  cosine similarity
idx = artists_list.index(selected_artist)
sims = cosine_similarity([scaled[idx]], scaled)[0]

sim_df = pd.DataFrame({
    "artist": artists_list,
    "similarity": sims
}).sort_values("similarity", ascending=False)

top5 = sim_df[sim_df["artist"] != selected_artist].head(5)

# show similar artists
st.subheader(f"Artists Similar to **{selected_artist}**")
st.table(top5.reset_index(drop=True))

# Radar chart
st.subheader("Audio Feature Profile")
values = artist_means.loc[selected_artist].values

fig = go.Figure(data=go.Scatterpolar(
    r=values,
    theta=artist_means.columns,
    fill='toself'
))
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)


# =======================================================
# 2) TREND SPOTTING
# =======================================================

st.header("Trend Spotting")

artist_tracks = df_small[df_small["artists"] == selected_artist]

col1, col2 = st.columns(2)
col1.metric("Tracks in Dataset", len(artist_tracks))
col2.metric("Average Popularity", round(artist_tracks["popularity"].mean(), 2))

st.subheader("Popularity Distribution")
fig = px.histogram(artist_tracks, x="popularity", nbins=20)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Audio Feature Distribution")
feat = st.selectbox("Feature", numeric_features)
fig = px.box(artist_tracks, y=feat)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Genre Occurrences")
fig = px.histogram(artist_tracks, x="track_genre")
st.plotly_chart(fig, use_container_width=True)


# ==========================================
# LOAD DATA
# ==========================================

df_playlist= pd.read_csv("Preprocessed_Data_CLEANED.csv")

st.header("Playlist Generator based on YOUR prefrance ")

# ==========================================
# USER INPUT SLIDERS
# ==========================================
st.subheader("Set Your Preferred Audio Ranges")

dance_min, dance_max = st.slider("Danceability", 0.0, 1.0, (0.3, 0.8))
energy_min, energy_max = st.slider("Energy", 0.0, 1.0, (0.3, 0.8))
valence_min, valence_max = st.slider("Valence (Happiness)", 0.0, 1.0, (0.2, 0.8))
tempo_min, tempo_max = st.slider("Tempo (BPM)", 40, 240, (80, 150))
acoustic_min, acoustic_max = st.slider("Acousticness", 0.0, 1.0, (0.0, 0.6))
instr_min, instr_max = st.slider("Instrumentalness", 0.0, 1.0, (0.0, 0.5))

top_n = st.selectbox("Number of Songs", [10, 20, 30, 50])


# ==========================================
# FILTERING LOGIC

if st.button("Generate Playlist"):
    filtered = df_playlist[
        (df_playlist["danceability"].between(dance_min, dance_max)) &
        (df_playlist["energy"].between(energy_min, energy_max)) &
        (df_playlist["valence"].between(valence_min, valence_max)) &
        (df_playlist["tempo"].between(tempo_min, tempo_max)) &
        (df_playlist["acousticness"].between(acoustic_min, acoustic_max)) &
        (df_playlist["instrumentalness"].between(instr_min, instr_max))
    ]


    filtered = filtered.sort_values(by="popularity", ascending=False).head(top_n)

    if filtered.empty:
        st.warning("No songs match your selected preferences. Try adjusting the ranges.")
    else:
        st.success(f"Showing {len(filtered)} songs")

        # nicer table
        st.dataframe(
            filtered[[
                "track_name", "artists", "track_genre", "popularity",
                "danceability", "energy", "valence", "tempo"
            ]]
        )

        # Save filtered for later use
        st.session_state["filtered_songs"] = filtered
# ==========================================
# --- SPOTIFY PLAYLIST CREATION ---


st.subheader("create Spotify Playlist Automatically")

playlist_name = st.text_input("Playlist Name", "made for you by you")

if st.button("Create Playlist in Spotify"):
    if "filtered_songs" not in st.session_state:
        st.error("Generate a playlist first!")
        st.stop()

    filtered = st.session_state["filtered_songs"]

    # Authenticate Spotify
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=st.secrets["client_id"],
        client_secret=st.secrets["client_secret"],
        redirect_uri=st.secrets["red_url"],
        scope="playlist-modify-public playlist-modify-private"
    ))

    user_id = sp.me()["id"]

    #  song names --> Spotify URIs
    st.write("Searching Spotify for each track…")
    uris = []
    for _, row in filtered.iterrows():
        query = f"track:{row['track_name']} artist:{row['artists'].split(';')[0]}"
        result = sp.search(query, type="track", limit=1)

        if result["tracks"]["items"]:
            uris.append(result["tracks"]["items"][0]["uri"])

    if len(uris) == 0:
        st.error("Could not find any songs on Spotify.")
        st.stop()

    #  playlist
    playlist = sp.user_playlist_create(
        user_id,
        playlist_name,
        public=True,
        description="generated for you
    )


    sp.playlist_add_items(playlist["id"], uris)

    st.success("playlist is readyyyyy")
    st.markdown(f"[Open in Spotify]({playlist['external_urls']['spotify']})")

