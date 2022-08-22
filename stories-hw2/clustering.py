from tslearn.clustering import TimeSeriesKMeans
import numpy as np
from sklearn.cluster import DBSCAN


def kmeans_cluster(data, metric='dtw', n_centers=5, seed=0, **kwargs):
    """
    :param data: Time series to cluster
    :param metric: defines distance between time serieses. Possible values: mae, cosine, dtw
    :param n_centers: Number of resulting clusters
    :param seed: random seed for reproducibility
    :return: list cluster indices, corresponding to time serieses
    """
    if metric == "cosine":
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        metric = "euclidean"
        clusterer = TimeSeriesKMeans(n_clusters=n_centers, metric=metric, random_state=seed)
    elif metric == "dtw":
        clusterer = TimeSeriesKMeans(n_clusters=n_centers, metric=metric, random_state=seed)
    elif metric == "mae":
        clusterer = DBSCAN(metric='l1', **kwargs)
    else:
        raise NotImplementedError()

    clusterer.fit(data)

    return clusterer.labels_
