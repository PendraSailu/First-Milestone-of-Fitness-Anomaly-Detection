from sklearn.cluster import KMeans, DBSCAN

def run_kmeans(X, clusters=2):
    model = KMeans(n_clusters=clusters, random_state=42)
    return model.fit_predict(X)

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)
