import plotly.graph_objects as go

def plot_time_series(df, time_col="timestamp", value_col="heart_rate"):
    """Plot the full time series."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[value_col],
        mode="lines+markers",
        name="Time Series",
        line=dict(color="blue")
    ))

    fig.update_layout(
        title="Time Series",
        xaxis_title="Time",
        yaxis_title=value_col,
        template="plotly_white"
    )

    return fig


def plot_anomalies(df, anomalies, time_col="timestamp", value_col="heart_rate"):
    """Plot full time series and highlight anomalies in red."""
    fig = go.Figure()

    # Full time series
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[value_col],
        mode="lines+markers",
        name="Data",
        line=dict(color="blue")
    ))

    # Anomalies
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies[time_col],
            y=anomalies[value_col],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=10, symbol="x")
        ))

    fig.update_layout(
        title="Anomaly Detection",
        xaxis_title="Time",
        yaxis_title=value_col,
        template="plotly_white"
    )

    return fig
