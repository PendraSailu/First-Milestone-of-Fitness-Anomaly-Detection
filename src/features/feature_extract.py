import pandas as pd
from tsfresh import extract_features

def extract_time_series_features(df, column_id="id", column_sort="timestamp"):
    return extract_features(df, column_id=column_id, column_sort=column_sort)
