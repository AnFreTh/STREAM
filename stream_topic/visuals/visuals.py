import matplotlib.pyplot as plt
from wordcloud import WordCloud

from ..models.abstract_helper_models.base import BaseModel, TrainingStatus
from ..utils import TMDataset
from ._interactive import (
    _visualize_topic_model_2d,
    _visualize_topic_model_3d,
    _visualize_topics_2d,
    _visualize_topics_3d,
)
from ._octis_visuals import OctisWrapperVisualModel


def visualize_topics_as_wordclouds(
    model,
    model_output: dict = None,
    max_words=50,
):
    """
    Visualize topics as word clouds.

    Args:
        model: Trained topic model.
        model_output (dict, optional): If visualizing an OCTIS model, pass the model_output as arguments
        max_words (int, optional): Maximum number of words to display in each word cloud (default is 100).

    Raises:
        AssertionError: If the model doesn't have the necessary output for topic visualization.

    Returns:
        None
            This function displays word clouds for each topic.
    """

    if not isinstance(model, BaseModel):
        if not model_output:
            raise TypeError(
                "If trying to visualize a non-Octis model, please pass the model_output"
            )
        print("--- Preparing Octis model for Visualizations ---")
        model = OctisWrapperVisualModel(model, model_output)
        model.get_topic_dict(top_words=max_words)

    assert (
        hasattr(model, "topic_dict") and model._status == TrainingStatus.SUCCEEDED
    ), "Model must have been trained with topics extracted."

    topics = model.topic_dict

    for topic_id, topic_words in topics.items():
        # Generate a word frequency dictionary for the topic
        word_freq = {word: weight for word, weight in topic_words}

        # Create and display the word cloud
        wordcloud = WordCloud(
            width=800, height=400, max_words=max_words, background_color="white"
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(f"Topic {topic_id}")
        plt.axis("off")
        plt.show()


def visualize_topic_model(
    model,
    model_output: dict = None,
    dataset: TMDataset = None,
    three_dim: bool = False,
    reduce_first: bool = False,
    reducer: str = "umap",
    port: int = 8050,
    embedding_model_name: str = "paraphrase-MiniLM-L3-v2",
    embeddings_folder_path: str = None,
    embeddings_file_path: str = None,
    use_average: bool = True,
):
    """
    Visualizes a topic model in 2D or 3D space, employing dimensionality reduction techniques such as UMAP, t-SNE, or PCA.
    This function facilitates an interactive exploration of topics and their associated documents or words.

    Parameters:
        model (AbstractModel): The trained topic model instance.
        model_output (dict, optional): The output of the topic model, typically including topic-word distributions and document-topic distributions. Required if the model does not have an 'output' attribute.
        dataset (TMDataset, optional): The dataset used for training the topic model. Required if the model does not have an 'output' attribute.
        three_dim (bool, optional): Flag to visualize in 3D if True, otherwise in 2D. Defaults to False.
        reduce_first (bool, optional): Indicates whether to perform dimensionality reduction on embeddings before computing topic centroids. Defaults to False.
        reducer (str, optional): Choice of dimensionality reduction technique. Supported values are 'umap', 'tsne', and 'pca'. Defaults to 'umap'.
        port (int, optional): The port number on which the visualization dashboard will run. Defaults to 8050.
        embedding_model_name (str, optional): Name of the embedding model used for generating document embeddings. Defaults to "all-MiniLM-L6-v2".
        embeddings_folder_path (str, optional): Path to the folder containing precomputed embeddings. If not provided, embeddings will be computed on the fly.
        embeddings_file_path (str, optional): Path to the file containing precomputed embeddings. If not provided, embeddings will be computed on the fly.

    Returns:
        None: Launches a Dash server that hosts the visualization dashboard, facilitating interactive exploration of the topic model.
    """

    if not isinstance(model, BaseModel):
        if not model_output:
            raise TypeError(
                "If trying to visualize an Octis model, please pass the model_output as well as the Dataset"
            )
        if not dataset:
            raise TypeError(
                "If trying to visualize an Octis model, please pass the model_output as well as the Dataset"
            )
        print("--- Preparing Octis model for Visualizations ---")
        model = OctisWrapperVisualModel(
            model,
            model_output,
            embedding_model_name,
            embeddings_folder_path,
            embeddings_file_path,
        )
        model.get_topic_dict(top_words=30)
        model.get_embeddings(dataset)

    assert (
        model._status == TrainingStatus.SUCCEEDED
    ), "Be sure to only pass a trained model to the visualization function"

    if three_dim:
        _visualize_topic_model_3d(
            model,
            reduce_first,
            reducer,
            port,
            dataset=dataset,
            encoder_model=embedding_model_name,
            use_average=use_average,
        )
    else:
        _visualize_topic_model_2d(
            model,
            reduce_first,
            reducer,
            port,
            dataset=dataset,
            encoder_model=embedding_model_name,
            use_average=use_average,
        )


def visualize_topics(
    model,
    model_output: dict = None,
    dataset: TMDataset = None,
    three_dim: bool = False,
    reducer: str = "umap",
    port: int = 8050,
    embedding_model_name: str = "paraphrase-MiniLM-L3-v2",
    embeddings_folder_path: str = None,
    embeddings_file_path: str = None,
    use_average: bool = True,
):
    """
    Visualize topics in either 2D or 3D space using UMAP, t-SNE, or PCA dimensionality reduction techniques.

    Args:
        model (AbstractModel): The trained topic model instance.
        model_output (dict, optional): The output of the topic model, typically including topic-word distributions and document-topic distributions. Required if the model does not have an 'output' attribute.
        dataset (TMDataset, optional): The dataset used for training the topic model. Required if the model does not have an 'output' attribute.
        three_dim (bool, optional): Flag to visualize in 3D if True, otherwise in 2D. Defaults to False.
        reduce_first (bool, optional): Indicates whether to perform dimensionality reduction on embeddings before computing topic centroids. Defaults to False.
        reducer (str, optional): Choice of dimensionality reduction technique. Supported values are 'umap', 'tsne', and 'pca'. Defaults to 'umap'.
        port (int, optional): The port number on which the visualization dashboard will run. Defaults to 8050.
        embedding_model_name (str, optional): Name of the embedding model used for generating document embeddings. Defaults to "all-MiniLM-L6-v2".
        embeddings_folder_path (str, optional): Path to the folder containing precomputed embeddings. If not provided, embeddings will be computed on the fly.
        embeddings_file_path (str, optional): Path to the file containing precomputed embeddings. If not provided, embeddings will be computed on the fly.


    Returns:
        None
            The function launches a Dash server to visualize the topic model.

    """
    if not isinstance(model, BaseModel):
        if not model_output:
            raise TypeError(
                "If trying to visualize an Octis model, please pass the model_output as well as the Dataset"
            )
        if not dataset:
            raise TypeError(
                "If trying to visualize an Octis model, please pass the model_output as well as the Dataset"
            )
        print("--- Preparing Octis model for Visualizations ---")
        model = OctisWrapperVisualModel(
            model,
            model_output,
            embedding_model_name,
            embeddings_folder_path,
            embeddings_file_path,
        )
        model.get_topic_dict(top_words=30)
        model.get_embeddings(dataset)

    assert (
        model._status == TrainingStatus.SUCCEEDED
    ), "Be sure to only pass a trained model to the visualization function"

    if three_dim:
        _visualize_topics_3d(
            model,
            reducer,
            port,
            dataset=dataset,
            encoder_model=embedding_model_name,
            use_average=use_average,
        )
    else:
        _visualize_topics_2d(
            model,
            reducer,
            port,
            dataset=dataset,
            encoder_model=embedding_model_name,
            use_average=use_average,
        )
