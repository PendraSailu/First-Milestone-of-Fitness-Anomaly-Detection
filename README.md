# 🏃‍♂️ FitPulse – Health Anomaly Detection from Fitness Devices  

FitPulse is a **health anomaly detection system** that processes fitness tracker data (heart rate, steps, sleep) to detect unusual patterns using **time-series analysis, machine learning, and clustering algorithms**.
It provides a **Streamlit-based interactive dashboard** for visualization and insights.  

## 📂 Project Structure  
<img width="213" height="853" alt="image" src="https://github.com/user-attachments/assets/610ac988-b5c9-4817-bb84-bf7e12c4fb43" />

## 📌 Features  

### 🔹 Data Collection & Preprocessing  
- Import fitness data (CSV/JSON).  
- Clean timestamps, fix missing values, align time intervals.  

### 🔹 Feature Extraction & Modeling  
- Extract statistical features with **TSFresh**.  
- Model seasonal patterns using **Facebook Prophet**.  
- Apply clustering algorithms (**KMeans, DBSCAN**) to group behaviors.  

### 🔹 Anomaly Detection & Visualization  
- Rule-based detection (thresholds).  
- Model-based detection (residuals, clustering outliers).  
- Interactive charts with **Matplotlib** and **Plotly**.  

### 🔹 Dashboard for Insights  
- Upload files and run anomaly detection.  
- View detected anomalies in real-time.  
- Export reports as **PDF/CSV**.  

## 🔎 Types of Anomalies  

- **Point Anomaly** → A single data point is unusual.  
  *Example: Heart rate spikes to 180 bpm while sitting.*  

- **Contextual Anomaly** → Data is abnormal in a specific context.  
  *Example: High heart rate is fine during running, but abnormal during sleep.*  

- **Collective Anomaly** → A sequence of points is unusual.  
  *Example: Sleep less than 3 hours continuously for 7 days.*
  
## 🛠️ Tools & Technology  

- **Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Plotly, Scikit-learn  
- **Feature Extraction**: TSFresh  
- **Time-Series Modeling**: Facebook Prophet  
- **Clustering**: KMeans, DBSCAN  
- **Dashboard**: Streamlit  
- **Data Formats**: CSV, JSON

## Create Virtual Environment
- python -m venv .venv
- .venv\Scripts\activate

## Install Dependencies
- pip install -r requirements.txt

## Run the Dashboard
- streamlit run src/app.py
  
## Upload Your Fitness Data
- Supported formats: CSV, JSON
- View detected anomalies and download reports.

## 📌 Future Improvements
- Support for real-time data streaming (IoT integration).
- Deep learning-based anomaly detection (LSTM/Autoencoders).
- Cloud deployment for wider accessibility.




