import pandas as pd

def split_to_sentences(text, splitter = "\n"):
    sentences = text.split(splitter)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def create_data_frame(sentences, pca_values, predictions):
    df = pd.DataFrame({
        "Sentence": sentences,
        "X": pca_values[:, 0],  # PCA Dimension 1
        "Y": pca_values[:, 1],  # PCA Dimension 2
        "Anomaly": ["Anomaly" if p == -1 else "Normal" for p in predictions],
    })
    return df