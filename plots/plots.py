import matplotlib.pyplot as plt
import mplcursors
import seaborn as sns
import plotly.express as px

def scatter_2d_plot(values, sentences):
    # Create a scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(values[:, 0], values[:, 1], color="blue", label="Sentences")
    # Add title and labels
    ax.set_title("2D Sentence Embeddings")
    #ax.set_xlabel("Dimension 1")
    #ax.set_ylabel("Dimension 2")

    # Use mplcursors to display sentences on hover
    cursor = mplcursors.cursor(scatter, hover=True)

    # Customize the tooltip to show the actual sentence
    @cursor.connect("add")
    def on_add(sel):
        # Get the index of the selected point
        index = sel.index
        # Set the annotation text to the corresponding sentence
        sel.annotation.set_text(sentences[index])
    plt.legend()
    plt.show()

def sentence_embedding_heatmap(embeddings,pca_values, sentences):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        embeddings,
        annot=False,  # Set to True to display values inside cells
        cmap="viridis",  # Choose a color map (e.g., "coolwarm", "magma", "plasma")
        xticklabels=[f"Dim {i + 1}" for i in range(pca_values.shape[1])],
        yticklabels=sentences,
    )

    # Add title and labels
    plt.title("Heatmap of Sentence Embeddings")
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Sentences")
    plt.show()

def similarity_heatmap(similarity,sentences):
    fig = px.imshow(
        similarity,
        text_auto=".3f",  # Annotate cells with similarity scores
        labels={"x": "Sentences", "y": "Sentences", "color": "Cosine Similarity"},
        x=sentences,  # X-axis labels
        y=sentences,  # Y-axis labels
        color_continuous_scale="turbo",  # Vibrant colormap
    )
    fig.update_layout(
        title="Similarity Heatmap",
        xaxis=dict(tickangle=-45),  # Rotate x-axis labels
        font=dict(size=10),  # Adjust font size
        autosize=False,
        width=1200,  # Width of the heatmap
        height=900,  # Height of the heatmap
    )
    fig.show()

def plot_anomaly(data_frame, color, algo):

    # Plotting with Plotly for interactive visualization
    fig = px.scatter(
        data_frame,
        x="X",
        y="Y",
        color=color,  # Color based on anomaly status
        hover_data={"Sentence": True, "X": False, "Y": False},
        title="Anomaly Detection in Text using " + algo,
        template="plotly_dark",  # Dark theme for better contrast
        color_discrete_map={"Anomaly": "red", "Normal": "green"},
    )

    # Customize layout
    fig.update_layout(
        legend_title="Anomaly Status",
        font=dict(size=12),
        width=900,
        height=600,
    )

    # Show the interactive plot
    fig.show()
