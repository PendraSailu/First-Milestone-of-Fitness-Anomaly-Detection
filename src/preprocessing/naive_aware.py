import pandas as pd

def fix_missing_values(df: pd.DataFrame, method="ffill") -> pd.DataFrame:
    """Fix missing values using forward fill or mean."""
    if method == "ffill":
        return df.fillna(method="ffill")
    elif method == "mean":
        return df.fillna(df.mean())
    else:
        return df
