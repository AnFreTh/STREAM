import matplotlib.pyplot as plt
from wordcloud import WordCloud
from ._interactive import (
    _visualize_topic_model_2d,
    _visualize_topic_model_3d,
    _visualize_topics_2d,
    _visualize_topics_3d,
)


def visualize_topics_as_wordclouds(model, max_words=100):
    """
    Visualize topics as word clouds.

    Args:
        model: Trained topic model.
        max_words (int, optional): Maximum number of words to display in each word cloud (default is 100).

    Raises:
        AssertionError: If the model doesn't have the necessary output for topic visualization.

    Returns:
        None
            This function displays word clouds for each topic.
    """

    assert (
        hasattr(model, "output") and "topic_dict" in model.output
    ), "Model must have been trained with topics extracted."

    topics = model.output["topic_dict"]

    for topic_id, topic_words in topics.items():
        # Generate a word frequency dictionary for the topic
        word_freq = {word: weight for word, weight in topic_words}

        # Create and display the word cloud
        wordcloud = WordCloud(
            width=800, height=400, max_words=max_words
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(f"Topic {topic_id}")
        plt.axis("off")
        plt.show()


def visualize_topic_model(
    model, three_dim=False, reduce_first=False, reducer="umap", port=8050
):
    """
    Visualize a topic model in either 2D or 3D space using UMAP, t-SNE, or PCA dimensionality reduction techniques.

    Args:
        model (AbstractModel): Trained topic model.
        three_dim (bool, optional): Whether to visualize in 3D or 2D (default is False).
        reduce_first (bool, optional): Whether to reduce dimensions of embeddings first and then compute topical centroids (default is False).
        reducer (str, optional): Dimensionality reduction technique to use, one of ['umap', 'tsne', 'pca'] (default is 'umap').
        port (int, optional): Port number for running the visualization dashboard (default is 8050).

    Returns:
        None
            The function launches a Dash server to visualize the topic model.
    """
    assert (
        model.trained
    ), "Be sure to only pass a trained model to the visualization function"

    if three_dim:
        _visualize_topic_model_3d(model, reduce_first, reducer, port)
    else:
        _visualize_topic_model_2d(model, reduce_first, reducer, port)


def visualize_topics(model, three_dim=False, reducer="umap", port=8050):
    """
    Visualize topics in either 2D or 3D space using UMAP, t-SNE, or PCA dimensionality reduction techniques.

    Args:
        model (AbstractModel): Trained topic model.
        three_dim (bool, optional): Whether to visualize in 3D or 2D (default is False).
        reducer (str, optional): Dimensionality reduction technique to use, one of ['umap', 'tsne', 'pca'] (default is 'umap').
        port (int, optional): Port number for running the visualization dashboard (default is 8050).

    Returns:
        None
            The function launches a Dash server to visualize the topic model.
    """

    assert (
        model.trained
    ), "Be sure to only pass a trained model to the visualization function"

    if three_dim:
        _visualize_topics_3d(model, reducer, port)
    else:
        _visualize_topics_2d(model, reducer, port)
