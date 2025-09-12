import json
import os
import pandas as pd

def load_json(path=None):
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(base_dir, "data", "sample_fitness_data.json")

    with open(path) as f:
        data = json.load(f)

    # Convert list/dict into DataFrame if possible
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame([data])
    else:
        raise ValueError("Unsupported JSON structure")

