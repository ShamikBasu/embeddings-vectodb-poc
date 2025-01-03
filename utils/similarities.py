from sklearn.metrics.pairwise import cosine_similarity

def similarity(embeddings):
    similarity = cosine_similarity(embeddings)
    return similarity