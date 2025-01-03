from plots.plots import scatter_2d_plot
from sentence_transformers import SentenceTransformer, util
from utils.PCA import PCA_VALUES
import torch

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed_using_paraphrase_MiniLM_L6_v2(sentences):
    embeddings = list(model.encode(sentences))
    values = PCA_VALUES(embeddings)
    return embeddings,values
    #scatter_2d_plot(values,sentences = sentences)


