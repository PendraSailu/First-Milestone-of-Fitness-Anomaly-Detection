import matplotlib.pyplot as plt

def plot_series(df, col="heart_rate", threshold=None):
    plt.figure(figsize=(8, 4))
    plt.plot(df["timestamp"], df[col], label=col)
    if threshold:
        plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
    plt.legend()
    plt.show()
