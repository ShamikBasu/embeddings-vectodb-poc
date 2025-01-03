from plots.plots import scatter_2d_plot
from sentence_transformers import SentenceTransformer, util
from utils.PCA import PCA_VALUES
import torch

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def embed_using_distilbert_base_nli_stsb_mean_tokens(sentences):
    embeddings = list(model.encode(sentences))
    values = PCA_VALUES(embeddings)
    return embeddings,values
    #scatter_2d_plot(values,sentences = sentences)


