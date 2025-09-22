from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd

def extract_tsfresh_features(df: pd.DataFrame, time_col="timestamp") -> pd.DataFrame:
    """Extract statistical features using TSFresh and clean them."""
    df = df.copy()
    df["id"] = 1  # single time series
    
    # Extract features
    features = extract_features(df, column_id="id", column_sort=time_col)
    
    # Clean NaN and inf values
    impute(features)
    
    return features
