import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.model import detect_anomalies, cluster_data
from features.feature_extract import extract_tsfresh_features

st.title("🏃‍♂️ FitPulse Health Anomaly Detection")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Fitness Data (CSV/JSON)", type=["csv", "json"])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # --- Rule-based anomalies ---
    st.subheader("Rule-based Anomalies")
    rule_anomalies = df[(df["heart_rate"] > 180) | (df["sleep_hours"] < 3)]
    st.dataframe(rule_anomalies if not rule_anomalies.empty else pd.DataFrame(columns=df.columns))

    # --- Residual anomalies (Prophet) ---
    st.subheader("Residual Anomalies")
    residuals = detect_anomalies(df)
    st.dataframe(residuals.head())

    # --- Time-series plot with anomalies ---
    st.subheader("📈 Heart Rate Over Time with Anomalies")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(residuals["timestamp"], residuals["heart_rate"], label="Heart Rate", color="blue")

    anomalies = residuals[residuals["residual"].abs() > 20]
    ax.scatter(anomalies["timestamp"], anomalies["heart_rate"], color="red", label="Anomaly")
    ax.set_title("Heart Rate Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Heart Rate")
    ax.legend()
    st.pyplot(fig)

    # --- Feature extraction + clustering ---
    features = extract_tsfresh_features(df)

    # Ensure features match original data length
    if features.shape[0] != len(df):
        st.warning(f"Features rows ({features.shape[0]}) do not match original data ({len(df)}). Filling missing rows with zeros.")
        missing_rows = len(df) - features.shape[0]
        if missing_rows > 0:
            filler = pd.DataFrame(0, index=range(missing_rows), columns=features.columns)
            features = pd.concat([features, filler], ignore_index=True)
        else:
            features = features.iloc[:len(df)]

    # Safe number of clusters
    n_clusters = min(5, len(df))

    # Get cluster labels
    labels = cluster_data(features, n_clusters=n_clusters)

    st.subheader("Cluster Labels")
    cluster_df = pd.DataFrame({"Cluster": labels})
    st.dataframe(cluster_df.value_counts().reset_index())

    # --- Cluster scatter plot ---
    st.subheader("🔹 Clusters (Steps vs Heart Rate)")
    cluster_plot_df = df.copy()
    cluster_plot_df["cluster"] = labels  # safe assignment

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        x="steps", y="heart_rate", hue="cluster",
        data=cluster_plot_df, palette="tab10", ax=ax2
    )
    ax2.set_title("Clusters of Activity")
    st.pyplot(fig2)
