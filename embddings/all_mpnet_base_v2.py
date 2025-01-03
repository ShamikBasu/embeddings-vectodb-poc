from plots.plots import scatter_2d_plot
from sentence_transformers import SentenceTransformer, util
from utils.PCA import PCA_VALUES
import torch

model = SentenceTransformer('all-mpnet-base-v2')

def embed_using_all_mpnet_base_v2(sentences):
    embeddings = list(model.encode(sentences))
    values = PCA_VALUES(embeddings)
    return embeddings,values
    #scatter_2d_plot(values,sentences = sentences)

