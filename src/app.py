import streamlit as st
import pandas as pd

st.title("Upload CSV Files")

uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")

all_dataframes = []

for uploaded_file in uploaded_files:
    try:
        # Try reading only the header first
        df_preview = pd.read_csv(uploaded_file, nrows=0)
        cols = df_preview.columns.tolist()

        # Determine if a timestamp column exists
        timestamp_col = None
        for col in cols:
            if col.lower() == "timestamp":
                timestamp_col = col
                break

        # Read the full CSV safely
        if timestamp_col:
            df = pd.read_csv(uploaded_file, parse_dates=[timestamp_col])
        else:
            df = pd.read_csv(uploaded_file)

        all_dataframes.append(df)
    
    except pd.errors.EmptyDataError:
        st.warning(f"Skipped '{uploaded_file.name}': file is empty.")
    except Exception as e:
        st.error(f"Error reading '{uploaded_file.name}': {e}")

# Combine all uploaded CSVs into a single DataFrame
if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    st.write("Combined DataFrame:")
    st.dataframe(combined_df)
else:
    st.info("No valid CSV files uploaded yet.")

