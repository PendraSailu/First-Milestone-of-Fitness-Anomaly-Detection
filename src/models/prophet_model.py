import pandas as pd
from prophet import Prophet

def train_prophet(df):
    df = df.rename(columns={"timestamp": "ds", "heart_rate": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    return model

def forecast(model, periods=10):
    future = model.make_future_dataframe(periods=periods, freq="H")
    return model.predict(future)
