from sklearn.decomposition import PCA

def PCA_VALUES(embeddings, n_components = 2):
    PCA_model = PCA(n_components=n_components)
    PCA_model.fit(embeddings)
    values = PCA_model.transform(embeddings)
    return values

def PCA_FOR_ANOMALY(embeddings):
    pca = PCA(n_components=2, random_state=2)
    embeddings_2d = pca.fit_transform(embeddings)
    return embeddings_2d