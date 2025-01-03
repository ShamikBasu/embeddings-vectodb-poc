from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM

import numpy as np
import pandas as pd
from utils.PCA import PCA_FOR_ANOMALY


def isolation_forest(embeddings, sentences, contamination =0.3):
    classifier = IsolationForest(contamination=contamination, random_state=2)  # Adjust contamination as needed
    predictions = classifier.fit_predict(embeddings)  # -1 = anomaly, 1 = normal
    # Output results
    for i, sentence in enumerate(sentences):
        label = "Anomaly" if predictions[i] == -1 else "Normal"
        print(f"Sentence: {sentence}")
        print(f"Prediction: {label}")
        print("-" * 50)
    #return predictions
    anomalies = predictions == -1
    embeddings_2d = PCA_FOR_ANOMALY(embeddings)
    data_frame = pd.DataFrame({
        "Sentence": sentences,
        "X": embeddings_2d[:, 0],  # PCA Dimension 1
        "Y": embeddings_2d[:, 1],  # PCA Dimension 2
        "Anomaly": ["Anomaly" if p == -1 else "Normal" for p in predictions],
    })
    return data_frame, anomalies


def k_means_clustering(embeddings, sentences, percentile = 90):
    # K-Means clustering
    n_clusters = 2  # We use 2 clusters: one for "normal" and one for "anomalies"
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Calculate the distance of each sentence to the cluster centers
    distances = kmeans.transform(embeddings).min(axis=1)
    threshold = np.percentile(distances, percentile)  # Top 1% distances considered as anomalies
    anomalies = distances > threshold  # Anomalous points are those with distance greater than the threshold
    embeddings_2d = PCA_FOR_ANOMALY(embeddings)
    data_frame = pd.DataFrame({
        "Sentence": sentences,
        "X": embeddings_2d[:, 0],  # PCA Dimension 1
        "Y": embeddings_2d[:, 1],  # PCA Dimension 2
        "Cluster": ["Anomaly" if is_anomaly else "Normal" for is_anomaly in anomalies],
        "Distance": distances,  # Distance to nearest cluster center
    })
    return data_frame

def dbscan_alg(embeddings,sentences, eps = 1.0):
    embeddings_2d = PCA_FOR_ANOMALY(embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=3)  # eps is the maximum distance between two samples to be considered as in the same neighborhood
    dbscan_preds = dbscan.fit_predict(embeddings)
    dbscan_anomalies = dbscan_preds == -1  # True for anomalies, False for normal points
    data_frame = pd.DataFrame({
        "Sentence": sentences,
        "X": embeddings_2d[:, 0],  # PCA Dimension 1
        "Y": embeddings_2d[:, 1],  # PCA Dimension 2
        "Cluster (DBSCAN)": ["Anomaly" if is_anomaly else "Normal" for is_anomaly in dbscan_anomalies],
    })
    return data_frame,dbscan_anomalies

def ocsvm(embeddings,sentences, nu=0.3):
    embeddings_2d = PCA_FOR_ANOMALY(embeddings)
    ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")  # nu is the proportion of outliers
    ocsvm_preds = ocsvm.fit_predict(embeddings)
    ocsvm_anomalies = ocsvm_preds == -1  # True for anomalies, False for normal points
    data_frame = pd.DataFrame({
        "Sentence": sentences,
        "X": embeddings_2d[:, 0],  # PCA Dimension 1
        "Y": embeddings_2d[:, 1],  # PCA Dimension 2
        "Cluster (One-Class SVM)": ["Anomaly" if is_anomaly else "Normal" for is_anomaly in ocsvm_anomalies],
    })
    return data_frame,ocsvm_anomalies



