import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """Load fitness data from CSV file."""
    return pd.read_csv(path)

