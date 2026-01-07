# 🎵 Music Popularity Prediction & Audio Pattern Discovery

**StreamFlow Music | AI & Data Science Project**

> Predicting song popularity, discovering audio patterns, and creating data-driven music experiences for 114,000+ Spotify tracks.

---

## 🚀 Project Overview
Music is more than sound—it's patterns, trends, and emotions. This project dives deep into **114,000 tracks** to uncover what makes a song popular, explore hidden music clusters, and build AI-powered tools for music curation and recommendation.

We combine **data preprocessing, EDA, supervised & unsupervised ML, deep learning**, and **interactive deployment** to transform raw Spotify data into actionable insights.

---

## 📊 Key Features

### 1️⃣ Data Preprocessing
- Handle missing audio features with **genre-based imputation**
- Create new insights from features:
  - **Energy/Danceability ratio**
  - **Acoustic ↔ Electronic spectrum**
- Normalize temporal features (e.g., `duration_ms`) using **log transformation**
- Encode high-cardinality variables (**artists, genres**)
- Correct **popularity bias** in training samples

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Popularity distribution across **decades & genres**
- **Correlation matrices** & **pair plots** of audio features
- Track **temporal trends** in music characteristics
- Analyze artist-level popularity patterns and career trajectories
- Study genre evolution and **crossover popularity effects**

---

### 3️⃣ Supervised ML
- Predict song popularity using **ensemble & neural methods**:
  - **XGBoost** with Bayesian hyperparameter tuning
  - **Random Forest** with feature importance
  - **Neural Networks** with embeddings for categorical features
- Evaluate using **business-relevant metrics** beyond RMSE
- Implement **temporal cross-validation** to prevent leakage

---

### 4️⃣ Unsupervised ML
- Discover **micro-genres** and hidden patterns:
  - **KMeans clustering (k=12)**
  - **DBSCAN** for outlier detection & niche music
  - **Hierarchical clustering** for taxonomy creation
- Dimensionality reduction with **PCA & t-SNE** for visualization
- Validate clusters with **musicological characteristics**

---

### 5️⃣ Deep Learning
- **DNN regressor** with 5 hidden layers for popularity prediction
- **Autoencoders** for unsupervised music representation learning
- Use **cluster embeddings** as additional model features
- Apply **attention mechanisms** for interpretable feature importance
- Compare deep learning results against traditional audio methods

---

### 6️⃣ Streamlit Deployment
- Interactive platform for **music analysis & recommendation**:
  - Input song features for **real-time popularity prediction** 🎶
  - Explore music clusters & find **similar tracks**
  - Visualize **genre evolution** across decades
  - Artist similarity & **trend spotting**
  - Generate **playlists based on audio preferences**

---

## 📂 Dataset
- **Spotify Tracks Dataset**
- **Size:** 114,000 tracks
- **Features:** 18 audio attributes
- **Target:** Popularity score (0–100)

---

## 🛠️ Tech Stack
- **Python:** pandas, numpy, scikit-learn, xgboost, tensorflow/keras
- **Visualization:** matplotlib, seaborn, plotly
- **Clustering & Dimensionality Reduction:** PCA, t-SNE, KMeans, DBSCAN
- **Deployment:** Streamlit

---

## ⚡ How to Run
```bash
# Clone repo
git clone https://github.com/yourusername/music-popularity.git

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
