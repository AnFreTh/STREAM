import torch
import numpy as np
from tqdm import tqdm
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
from ..utils.tf_idf import c_tf_idf, extract_tfidf_topics
from ..data_utils.dataset import TMDataset
import numpy as np
from itertools import product
import umap.umap_ as umap
from sklearn.preprocessing import OneHotEncoder


class SOMTM(AbstractModel):
    def __init__(
        self,
        m: int = None,
        n: int = None,
        dim: int = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        n_iterations: int = 100,
        batch_size: int = 128,
        lr: float = None,
        sigma: float = None,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        reduce_dim: bool = True,
        reduced_dimension: int = 16,
        umap_args: dict = {},
        use_softmax: bool = True,
    ):
        """
        Initialize the Self-Organizing Map for Topic modeling (SOMTM) model.

        Parameters:
            m (int, optional): Number of rows in the SOM grid (default is None).
            n (int, optional): Number of columns in the SOM grid (default is None).
            dim (int, optional): Dimensionality of the training inputs (default is None).
            embedding_model_name (str, optional): Name of the SentenceTransformer embedding model (default is "all-MiniLM-L6-v2").
            n_iterations (int, optional): Number of iterations for training (default is 100).
            batch_size (int, optional): Batch size for training (default is 128).
            lr (float, optional): Initial learning rate (default is None, which sets it to 0.3).
            sigma (float, optional): Initial neighborhood value (default is None, which sets it to max(m, n) / 2).
            embeddings_folder_path (str, optional): Path to the folder containing precomputed embeddings (default is None).
            embeddings_file_path (str, optional): Path to the precomputed embeddings file (default is None).
            reduce_dim (bool, optional): Whether to reduce dimensionality (default is True).
            reduced_dimension (int, optional): Reduced dimensionality (default is 16).
            umap_args (dict, optional): Arguments for UMAP dimensionality reduction (default is {}).
            use_softmax (bool, optional): Whether to use softmax for mapping (default is True).

        """

        super().__init__()
        self.trained = False
        self.m = m
        self.n = n
        if reduce_dim:
            self.dim = reduced_dimension
        else:
            self.dim = dim
        self.n_iterations = n_iterations
        self.alpha = lr if lr is not None else 0.3
        self.sigma = sigma if sigma is not None else max(m, n) / 2
        self.batch_size = batch_size
        self.use_softmax = use_softmax

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path
        self.reduce_dim = reduce_dim

        # Initialize weight vectors for each neuron
        self.weights = torch.randn(m * n, self.dim)
        self.locations = torch.tensor(list(product(range(m), range(n))))
        self.train_history = []

        self.umap_args = (
            umap_args
            if umap_args
            else {
                "n_neighbors": 15,
                "n_components": self.dim,
                "metric": "cosine",
            }
        )

    def _prepare_data(self):
        """
        Prepares the dataset for clustering.

        """

        self.embeddings = self.dataset.get_embeddings(
            self.embedding_model_name, self.embeddings_path, self.embeddings_file_path
        )
        self.dataframe = self.dataset.dataframe

    def _find_bmu(self, x):
        """
        Find the Best Matching Unit (BMU) for a given vector x
        """
        differences = self.weights - x
        distances = torch.sum(torch.pow(differences, 2), 1)
        min_index = torch.argmin(distances)
        return min_index

    def _decay_learning_rate(self, iteration):
        """
        Decay the learning rate over time
        """
        return self.alpha * (1 - (iteration / self.n_iterations))

    def _decay_radius(self, iteration):
        """
        Decay the neighborhood radius over time
        """
        return self.sigma * np.exp(-iteration / self.n_iterations)

    def _update_weights_batch(self, batch, bmu_indices, iteration):
        lr = self._decay_learning_rate(iteration)
        rad = self._decay_radius(iteration)
        rad_squared = rad**2

        for i, x in enumerate(batch):
            bmu = bmu_indices[i]
            distance_squares = torch.sum(
                torch.pow(self.locations - self.locations[bmu], 2), 1
            )

            if self.use_softmax:
                influence = torch.nn.functional.softmax(
                    -distance_squares / (2 * rad_squared)
                )
            else:
                influence = torch.exp(-distance_squares / (2 * rad_squared))

            influence = influence.unsqueeze(1)
            self.weights += lr * influence * (x - self.weights)

    def _train_batch(self, data, batch_size):
        """
        Train the SOM using mini-batches.
        """
        n_samples = len(data)
        for iteration in tqdm(range(self.n_iterations)):
            # Shuffle data at each epoch
            np.random.shuffle(data)

            for i in range(0, n_samples, batch_size):
                batch = data[i : i + batch_size]
                batch_tensor = torch.tensor(batch)
                bmu_indices = [self._find_bmu(x) for x in batch_tensor]

                self._update_weights_batch(batch_tensor, bmu_indices, iteration)

        self.labels = self._get_cluster_labels(data)

    def _dim_reduction(self):
        """
        Reduces the dimensionality of embeddings using UMAP.
        """
        try:
            self.reducer = umap.UMAP(**self.umap_args)
            self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
        except Exception as e:
            raise ValueError(f"Error in dimensionality reduction: {e}")

    def _get_weights(self):
        """
        Get the trained weights of the SOM
        """
        return self.weights

    def _get_cluster_labels(self, data):
        """
        Assigns each data point to the closest cluster (BMU).
        :param data: Iterable of data points.
        :return: List of cluster indices corresponding to each data point.
        """
        labels = []
        for x in data:
            x_tensor = torch.tensor(x)
            bmu_index = self._find_bmu(x_tensor)
            labels.append(bmu_index.item())  # Convert tensor to integer
        return labels

    def _get_topic_document_matrix(self):
        assert (
            self.trained
        ), "Model must be trained before accessing the topic-document matrix."
        # Safely get the topic-document matrix with a default value of None if not found
        return self.output.get("topic-document-matrix", None)

    def train_model(self, dataset, n_top_words: int = 10):
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
        if self.reduce_dim:
            print("--- Dimensionality Reduction ---")
            self._dim_reduction()
            print("--- start training ---")
            self._train_batch(self.reduced_embeddings, self.batch_size)
        else:
            print("--- start training ---")
            self._train_batch(self.embeddings, self.batch_size)

        self.dataframe["predictions"] = self.labels
        docs_per_topic = self.dataframe.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )

        print("--- Extracting the Topics ---")
        tfidf, count = c_tf_idf(docs_per_topic["text"].values, m=len(self.dataframe))
        topics = extract_tfidf_topics(tfidf, count, docs_per_topic, n=n_top_words)

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
