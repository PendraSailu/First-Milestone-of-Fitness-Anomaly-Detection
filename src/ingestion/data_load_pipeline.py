from src.ingestion.load_csv import load_csv
from src.ingestion.load_json import load_json


def load_all():
    csv_data = load_csv("data/sample_heart_rate.csv")
    json_data = load_json("data/sample_fitness_data.json")
    return csv_data, json_data

if __name__ == "__main__":
    c, j = load_all()
    print("CSV sample:\n", c.head())
    print("\nJSON sample:\n", j.head())
