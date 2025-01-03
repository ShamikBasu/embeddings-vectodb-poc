from utils.data_utils import split_to_sentences
from embddings.all_MiniLM_L6_v2 import embed_using_all_MiniLM_L6_v2
from embddings.all_mpnet_base_v2 import embed_using_all_mpnet_base_v2
from embddings.paraphrase_MiniLM_L6_v2 import embed_using_paraphrase_MiniLM_L6_v2
from embddings.distilbert_base_nli_stsb_mean_tokens import embed_using_distilbert_base_nli_stsb_mean_tokens
from plots.plots import scatter_2d_plot,sentence_embedding_heatmap,similarity_heatmap,plot_anomaly
from utils.similarities import similarity
from machine_learning.anomaly_detection import isolation_forest,dbscan_alg,ocsvm
text = """
    Preheat the oven to the correct temperature for the desired cake.
    Grease and flour the baking pan to prevent sticking.
    In a separate bowl, whisk together the dry ingredients: flour, sugar, baking powder, and salt.
    In another bowl, beat together the wet ingredients: eggs, milk, and melted butter.
    Gradually add the wet ingredients to the dry ingredients, mixing until just combined.
    Pour the batter into the prepared pan and bake until a toothpick inserted into the center comes out clean.
    The majestic lion surveyed its kingdom from atop a towering rock.
    Allow the cake to cool completely before frosting.
    A mermaid emerged from the ocean, her shimmering tail glinting in the sunlight.
    Frost the cake with your favorite frosting and decorate as desired.
    The tiny spaceship zoomed through the galaxy, leaving a trail of sparkling stardust.
"""
# Text to array of sentences
sentences = split_to_sentences(text)

#embed the sentences
embeddings, pca_values = embed_using_all_MiniLM_L6_v2(sentences)
#embeddings, pca_values = embed_using_distilbert_base_nli_stsb_mean_tokens(sentences)
#embeddings, pca_values = embed_using_all_mpnet_base_v2(sentences)
#embeddings, pca_values = embed_using_paraphrase_MiniLM_L6_v2(sentences)

#plot the embeddings
scatter_2d_plot(pca_values,sentences)
sentence_embedding_heatmap(embeddings,pca_values,sentences)

# find similarity
similarities = similarity(embeddings)
similarity_heatmap(similarities,sentences)

#anomaly
data_frame_isolation, anomalies_isolation = isolation_forest(embeddings,sentences,contamination=0.3)
plot_anomaly(data_frame_isolation,"Anomaly", "Isolation Forest")

anomalous_sentences = [sentence for sentence, is_anomaly in zip(sentences, anomalies_isolation) if is_anomaly]
print("Anomalous Sentences Detected ISOLATION FOREST:")
for sentence in anomalous_sentences:
    print(f"- {sentence}")

data_frame_dbscan, anomalies_dbscan = dbscan_alg(embeddings,sentences,eps=1.0)
plot_anomaly(data_frame_dbscan, "Cluster (DBSCAN)", "DBSCAN")

anomalous_sentences = [sentence for sentence, is_anomaly in zip(sentences, anomalies_dbscan) if is_anomaly]
print("Anomalous Sentences Detected DBSCAN:")
for sentence in anomalous_sentences:
    print(f"- {sentence}")

data_frame_ocsvm, anomalies_ocsvm = ocsvm(embeddings,sentences,nu=0.3)
plot_anomaly(data_frame_ocsvm,"Cluster (One-Class SVM)", "One Class - SVM")
anomalous_sentences = [sentence for sentence, is_anomaly in zip(sentences, anomalies_ocsvm) if is_anomaly]
print("Anomalous Sentences Detected One-Class SVM:")
for sentence in anomalous_sentences:
    print(f"- {sentence}")