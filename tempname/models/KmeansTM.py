from sklearn.cluster import KMeans
import umap.umap_ as umap
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ..utils.tf_idf import c_tf_idf, extract_tfidf_topics
from ..data_utils.dataset import TMDataset
import numpy as np


class KmeansTM(AbstractModel):
    """
    A topic modeling class that uses K-Means clustering on text data.

    This class inherits from the AbstractModel class and utilizes sentence embeddings,
    UMAP for dimensionality reduction, and K-Means for clustering text data into topics.

    Attributes:
        hyperparameters (dict): A dictionary of hyperparameters for the model.
        n_topics (int): The number of topics to cluster the documents into.
        embedding_model (SentenceTransformer): The sentence embedding model.
        umap_args (dict): Arguments for UMAP dimensionality reduction.
        kmeans_args (dict): Arguments for the KMeans clustering algorithm.
        optim (bool): Flag to enable optimization of the number of clusters.
    """

    def __init__(
        self,
        num_topics: int = 20,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        umap_args: dict = {},
        kmeans_args: dict = {},
        random_state: int = None,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
    ):
        """
        Initializes the KmeansTM model with specified parameters.

        Parameters:
            hyperparameters (dict): Model hyperparameters. Defaults to an empty dict.
            num_topics (int): Number of topics. Defaults to 20.
            embedding_model (SentenceTransformer): Sentence embedding model. Defaults to "all-MiniLM-L6-v2".
            umap_args (dict): UMAP arguments. Defaults to an empty dict.
            kmeans_args (dict): KMeans arguments. Defaults to an empty dict.
            random_state (int): Random state for reproducibility. Defaults to None.
        """
        super().__init__()
        self.trained = False
        self.n_topics = num_topics
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.umap_args = (
            umap_args
            if umap_args
            else {
                "n_neighbors": 15,
                "n_components": 15,
                "metric": "cosine",
            }
        )
        if random_state is not None:
            self.umap_args["random_state"] = random_state
        self.kmeans_args = kmeans_args
        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path

        assert (
            isinstance(num_topics, int) and num_topics > 0
        ), "num_topics must be a positive integer."

    def _prepare_data(self):
        """
        Prepares the dataset for clustering.

        """

        self.embeddings = self.dataset.get_embeddings(
            self.embedding_model_name, self.embeddings_path, self.embeddings_file_path
        )
        self.dataframe = self.dataset.dataframe

    def _clustering(self):
        """
        Applies K-Means clustering to the reduced embeddings.
        """
        try:
            clustering_model = KMeans(n_clusters=self.n_topics, **self.kmeans_args)
            clustering_model.fit(self.reduced_embeddings)
            self.labels = clustering_model.labels_

        except Exception as e:
            raise ValueError(f"Error in clustering: {e}")

        labels = np.array(self.labels)

        # Initialize an empty dictionary to store mean embeddings for each label
        self.topic_centroids = []

        # Iterate over unique labels and compute mean embedding for each
        for label in np.unique(labels):
            # Find embeddings corresponding to the current label
            label_embeddings = self.embeddings[labels == label]

            # Compute mean embedding for the current label
            mean_embedding = np.mean(label_embeddings, axis=0)

            # Store the mean embedding in the dictionary
            self.topic_centroids.append(mean_embedding)

    def _dim_reduction(self):
        """
        Reduces the dimensionality of embeddings using UMAP.
        """
        try:
            self.reducer = umap.UMAP(**self.umap_args)
            self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
        except Exception as e:
            raise ValueError(f"Error in dimensionality reduction: {e}")

    def _get_topic_document_matrix(self):
        assert (
            self.trained
        ), "Model must be trained before accessing the topic-document matrix."
        # Safely get the topic-document matrix with a default value of None if not found
        return self.output.get("topic-document-matrix", None)

    def train_model(self, dataset):
        """
        Trains the K-Means topic model on the provided dataset.

        Applies sentence embedding, UMAP dimensionality reduction, and K-Means clustering
        to the dataset to identify distinct topics within the text data.

        Parameters:
            dataset: The dataset to train the model on. It should contain the text documents.

        Returns:
            dict: A dictionary containing the identified topics and the topic-word matrix.
        """

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."
        self.dataset = dataset
        print("--- preparing the dataset ---")
        self._prepare_data()
        print("--- Dimensionality Reduction ---")
        self._dim_reduction()
        print("--- Training the model ---")
        self._clustering()

        self.dataframe["predictions"] = self.labels
        docs_per_topic = self.dataframe.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )

        print("--- Extracting the Topics ---")
        tfidf, count = c_tf_idf(docs_per_topic["text"].values, m=len(self.dataframe))
        topics = extract_tfidf_topics(tfidf, count, docs_per_topic, n=10)

        one_hot_encoder = OneHotEncoder(
            sparse=False
        )  # Use sparse=False to get a dense array
        predictions_one_hot = one_hot_encoder.fit_transform(
            self.dataframe[["predictions"]]
        )

        # Transpose the one-hot encoded matrix to get shape (k, n)
        topic_document_matrix = predictions_one_hot.T

        self.output = {
            "topics": [[word for word, _ in topics[key]] for key in topics],
            "topic-word-matrix": tfidf.T,
            "topic_dict": topics,
            "topic-document-matrix": topic_document_matrix,  # Include the transposed one-hot encoded matrix
        }
        self.trained = True
        return self.output
