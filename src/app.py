import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.model import detect_anomalies, cluster_data
from features.feature_extract import extract_tsfresh_features

# Custom CSS for Light Skyblue Dashboard 
st.markdown("""
<style>
/* Page background */
body {
    background-color: #f0f8ff;  /* very light sky blue */
    font-family: 'Poppins', sans-serif;
}

/* Title styling */
h1 {
    color: #0c4a6e;  /* muted dark blue */
    font-size: 2.5rem;
    text-align: center;
    font-weight: 700;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #bae6fd;  /* soft sky blue */
    color: #0c4a6e;  /* dark text */
    padding: 20px;
    border-radius: 12px;
    font-weight: 600;
}

/* Sidebar headers */
[data-testid="stSidebar"] h2 {
    color: #0c4a6e;
}

/* Sidebar mini cards */
.sidebar-card {
    background-color: #7dd3fc;  /* slightly darker sky blue */
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    color: #0c4a6e;
    text-align: center;
    font-weight: 600;
    transition: transform 0.2s;
}
.sidebar-card:hover {
    transform: scale(1.02);
    cursor: pointer;
}

/* Metrics cards styling */
.css-1v3fvcr, .stMetric {
    background-color: #7dd3fc;  
    color: #0c4a6e;
    border-radius: 10px;
    padding: 10px 15px;
    font-weight: bold;
    text-align: center;
}

/* Tabs styling */
.stTabs [role="tab"] {
    font-weight: bold;
    font-size: 1rem;
    background-color: #e0f2fe;  /* light sky blue for inactive */
    color: #0c4a6e;
    border-radius: 10px 10px 0 0;
    margin-right: 5px;
    padding: 10px;
}
.stTabs [role="tab"].st-bz {
    background-color: #38bdf8;  /* muted sky blue active tab */
    color: #ffffff;
}

/* Dataframes styling */
.stDataFrame table {
    border-radius: 10px;
    border-collapse: separate !important;
    border-spacing: 0 10px;
    background-color: #ffffff;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton button {
    background-color: #7dd3fc;  
    color: #0c4a6e;
    font-weight: bold;
    border-radius: 10px;
    padding: 8px 15px;
}
</style>
""", unsafe_allow_html=True)

# Page Config 
st.set_page_config(page_title="FitPulse Health Anomaly Detection", page_icon="🏃‍♂️", layout="centered")
st.title("🏃‍♂️ FitPulse Health Anomaly Detection")

# Sidebar 
st.sidebar.header("📂 Upload & Quick Actions")

# File uploader at top
uploaded_file = st.sidebar.file_uploader("📤 Browse your fitness file (CSV/JSON)", type=["csv", "json"])

# Sidebar legend card
st.sidebar.markdown(
    """
    <div class="sidebar-card">
    <strong>ℹ️ Tips & Legend</strong><br>
    - Heart rate > 180 → risky pulse<br>
    - Sleep hours < 3 → poor rest<br>
    - Residual > 20 → unusual change
    </div>
    """, unsafe_allow_html=True
)

# Main content 
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    # Detect anomalies
    rule_anomalies = df[(df["heart_rate"] > 180) | (df["sleep_hours"] < 3)]
    residuals = detect_anomalies(df)
    residual_anomalies = residuals[residuals["residual"].abs() > 20]
    total_anomalies = len(rule_anomalies) + len(residual_anomalies)

    # Sidebar quick stats
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Total Records", len(df))
    st.sidebar.metric("⚠️ Total Anomalies", total_anomalies)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "⚠️ Anomalies", "🔹 Clustering"])

    # Tab 1: Raw Data 
    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

    # Tab 2: Anomalies
    with tab2:
        st.subheader("Anomaly Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rule-based", len(rule_anomalies))
        c2.metric("Residual", len(residual_anomalies))
        c3.metric("Total", total_anomalies)

        st.subheader("Rule-based Anomalies")
        st.dataframe(rule_anomalies if not rule_anomalies.empty else pd.DataFrame(columns=df.columns))

        st.subheader("Residual Anomalies")
        st.dataframe(residuals.head())

        # Heart Rate Plot in Red 
        st.subheader("📈 Heart Rate Over Time with Anomalies")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor("#f0f8ff")
        ax.grid(color="#e0f2fe", linestyle="--", linewidth=0.5)
        ax.plot(residuals["timestamp"], residuals["heart_rate"], label="Heart Rate", color="red", linewidth=2)
        ax.scatter(residual_anomalies["timestamp"], residual_anomalies["heart_rate"], color="#8B0000", s=60, label="Anomaly", edgecolor="black", zorder=5)
        ax.set_title("Heart Rate Over Time", fontsize=16, color="#0c4a6e")
        ax.set_xlabel("Timestamp", fontsize=12)
        ax.set_ylabel("Heart Rate", fontsize=12)
        ax.legend(facecolor="#7dd3fc")
        st.pyplot(fig)

        st.download_button("⬇️ Download Anomalies (CSV)", pd.concat([rule_anomalies, residual_anomalies]).to_csv(index=False), file_name="anomalies.csv")

    # Tab 3: Clustering 
    with tab3:
        st.subheader("Feature Extraction & Clustering")
        features = extract_tsfresh_features(df)

        if features.shape[0] != len(df):
            missing_rows = len(df) - features.shape[0]
            filler = pd.DataFrame(0, index=range(missing_rows), columns=features.columns)
            features = pd.concat([features, filler], ignore_index=True)

        n_clusters = min(5, len(df))
        labels = cluster_data(features, n_clusters=n_clusters)

        st.subheader("Cluster Labels")
        cluster_df = pd.DataFrame({"Cluster": labels})
        st.dataframe(cluster_df.value_counts().reset_index())

        st.subheader("🔹 Clusters (Steps vs Heart Rate)")
        cluster_plot_df = df.copy()
        cluster_plot_df["cluster"] = labels

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.set_facecolor("#f0f8ff")
        ax2.grid(color="#e0f2fe", linestyle="--", linewidth=0.5)
        palette = sns.color_palette("Set1", n_colors=n_clusters)  # more distinct colors
        sns.scatterplot(x="steps", y="heart_rate", hue="cluster", data=cluster_plot_df, palette=palette, s=80, edgecolor="black", ax=ax2)
        ax2.set_title("Clusters of Activity", fontsize=16, color="#0c4a6e")
        ax2.set_xlabel("Steps", fontsize=12)
        ax2.set_ylabel("Heart Rate", fontsize=12)
        ax2.legend(title="Cluster", facecolor="#7dd3fc", edgecolor="#0c4a6e")
        st.pyplot(fig2)
