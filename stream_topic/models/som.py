from datetime import datetime
from itertools import product

import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from ..commons.check_steps import check_dataset_steps
from ..preprocessor._tf_idf import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.mixins import SentenceEncodingMixin

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "SOMTM"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class SOMTM(BaseModel, SentenceEncodingMixin):
    def __init__(
        self,
        m: int,
        n: int,
        umap_args: dict = {},
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        reduce_dim: bool = True,
        reduced_dimension: int = 16,
        dim: int = None,
        **kwargs,
    ):
        """
        Initialize the Self-Organizing Map for Topic Modeling (SOMTM) model.

        Parameters
        ----------
        m : int
            Number of rows in the SOM grid.
        n : int
            Number of columns in the SOM grid.
        umap_args : dict, optional
            Arguments for UMAP dimensionality reduction (default is {}).
        embedding_model_name : str, optional
            Name of the SentenceTransformer embedding model (default is "paraphrase-MiniLM-L3-v2").
        embeddings_folder_path : str, optional
            Path to the folder containing precomputed embeddings (default is None).
        embeddings_file_path : str, optional
            Path to the precomputed embeddings file (default is None).
        save_embeddings : bool, optional
            Whether to save embeddings (default is False).
        reduce_dim : bool, optional
            Whether to reduce dimensionality (default is True).
        reduced_dimension : int, optional
            Reduced dimensionality (default is 16).
        dim : int, optional
            Dimensionality of the training inputs (default is None).
        kwargs : dict
            Additional arguments.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(
            ignore=[
                "embeddings_file_path",
                "embeddings_folder_path",
                "random_state",
                "save_embeddings",
            ]
        )

        self.m = self.hparams.get("m", m)
        self.n = self.hparams.get("n", n)

        self.embedding_model_name = self.hparams.get(
            "embedding_model_name", embedding_model_name
        )
        self.embeddings_path = self.hparams.get(
            "embeddings_folder_path", embeddings_folder_path
        )
        self.embeddings_file_path = self.hparams.get(
            "embeddings_file_path", embeddings_file_path
        )

        # Initialize weight vectors for each neuron
        self.reduce_dim = self.hparams.get("reduce_dim", reduce_dim)
        self.dim = (
            self.hparams.get("reduced_dimension", reduced_dimension)
            if self.reduce_dim
            else dim
        )
        self.weights = torch.randn(self.m * self.n, self.dim)
        self.locations = torch.tensor(
            list(product(range(self.m), range(self.n))))
        self.train_history = []

        self.umap_args = self.hparams.get(
            "umap_args",
            umap_args
            or {
                "n_neighbors": 15,
                "n_components": reduced_dimension,
                "metric": "cosine",
            },
        )

        self.save_embeddings = save_embeddings
        self._status = TrainingStatus.NOT_STARTED

    def get_info(self):
        """
        Get information about the model.

        Returns
        -------
        dict
            Dictionary containing model information including model name,
            number of topics, embedding model name, UMAP arguments,
            K-Means arguments, and training status.
        """
        info = {
            "model_name": MODEL_NAME,
            "num_topics": np.round(self.m * self.n),
            "embedding_model": self.embedding_model_name,
            "umap_args": self.umap_args,
            "trained": self._status.name,
            "hyperparameters": self.hparams,
        }
        return info

    def _find_bmu(self, x):
        """
        Find the Best Matching Unit (BMU) for a given vector.

        Parameters
        ----------
        x : torch.Tensor
            Input vector.

        Returns
        -------
        int
            Index of the BMU.
        """
        differences = self.weights - x
        distances = torch.sum(torch.pow(differences, 2), 1)
        min_index = torch.argmin(distances)
        return min_index

    def _decay_learning_rate(self, iteration):
        """
        Decay the learning rate over time.

        Parameters
        ----------
        iteration : int
            Current iteration.

        Returns
        -------
        float
            Decayed learning rate.
        """
        return self.alpha * (1 - (iteration / self.n_iterations))

    def _decay_radius(self, iteration):
        """
        Decay the neighborhood radius over time.

        Parameters
        ----------
        iteration : int
            Current iteration.

        Returns
        -------
        float
            Decayed neighborhood radius.
        """
        return self.sigma * np.exp(-iteration / self.n_iterations)

    def _update_weights_batch(self, batch, bmu_indices, iteration):
        """
        Update the weight vectors for a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            Batch of input vectors.
        bmu_indices : list of int
            List of BMU indices for the batch.
        iteration : int
            Current iteration.
        """
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
                    -distance_squares / (2 * rad_squared), dim=0
                )
            else:
                influence = torch.exp(-distance_squares / (2 * rad_squared))

            influence = influence.unsqueeze(1)
            self.weights += lr * influence * (x - self.weights)

    def _train_batch(self, data, batch_size):
        """
        Train the SOM using mini-batches.

        Parameters
        ----------
        data : numpy.ndarray
            Training data.
        batch_size : int
            Size of each mini-batch.
        """
        n_samples = len(data)
        for iteration in tqdm(range(self.n_iterations)):
            # Shuffle data at each epoch
            np.random.shuffle(data)

            for i in range(0, n_samples, batch_size):
                batch = data[i: i + batch_size]
                batch_tensor = torch.tensor(batch)
                bmu_indices = [self._find_bmu(x) for x in batch_tensor]

                self._update_weights_batch(
                    batch_tensor, bmu_indices, iteration)

        self.labels = self._get_cluster_labels(data)

    def _get_weights(self):
        """
        Get the trained weights of the SOM.

        Returns
        -------
        torch.Tensor
            Trained weight vectors.
        """
        return self.weights

    def _get_cluster_labels(self, data):
        """
        Assigns each data point to the closest cluster (BMU).

        Parameters
        ----------
        data : numpy.ndarray
            Input data points.

        Returns
        -------
        list of int
            List of cluster indices corresponding to each data point.
        """
        labels = []
        for x in data:
            x_tensor = torch.tensor(x)
            bmu_index = self._find_bmu(x_tensor)
            labels.append(bmu_index.item())  # Convert tensor to integer
        return labels

    def fit(
        self,
        dataset: TMDataset = None,
        n_iterations: int = 100,
        batch_size: int = 128,
        lr: float = None,
        sigma: float = None,
        use_softmax: bool = True,
    ):
        """
        Fit the SOMTM model to the dataset.

        Parameters
        ----------
        dataset : TMDataset, optional
            The dataset to fit the model to.
        n_iterations : int, optional
            Number of iterations for training (default is 100).
        batch_size : int, optional
            Batch size for training (default is 128).
        lr : float, optional
            Initial learning rate (default is None, which sets it to 0.3).
        sigma : float, optional
            Initial neighborhood value (default is None, which sets it to max(m, n) / 2).
        use_softmax : bool, optional
            Whether to use softmax for mapping (default is True).
        """

        self.n_iterations = n_iterations
        self.alpha = lr if lr is not None else 0.3
        self.sigma = sigma if sigma is not None else max(self.m, self.n) / 2
        self.batch_size = batch_size
        self.use_softmax = use_softmax

        self.hparams.update(
            {
                "n_iterations": n_iterations,
                "batch_size": batch_size,
                "lr": self.alpha,
                "sigma": self.sigma,
                "use_softmax": use_softmax,
            }
        )

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        check_dataset_steps(dataset, logger, MODEL_NAME)

        self._status = TrainingStatus.INITIALIZED
        try:
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            self.dataframe, self.embeddings = self.prepare_embeddings(
                dataset, logger)

            if self.reduce_dim:
                self.reduced_embeddings = self.dim_reduction(logger)
                self._train_batch(self.reduced_embeddings, self.batch_size)
            else:
                self._train_batch(self.embeddings, self.batch_size)
            self.dataframe["predictions"] = self.labels
            docs_per_topic = self.dataframe.groupby(
                ["predictions"], as_index=False
            ).agg({"text": " ".join})

            tfidf, count = c_tf_idf(
                docs_per_topic["text"].values, m=len(self.dataframe)
            )
            self.topic_dict = extract_tfidf_topics(
                tfidf, count, docs_per_topic, n=100)

            one_hot_encoder = OneHotEncoder(sparse=False)
            predictions_one_hot = one_hot_encoder.fit_transform(
                self.dataframe[["predictions"]]
            )
            self.beta = tfidf.T
            self.theta = predictions_one_hot.T
        except Exception as e:
            logger.error(f"Error in training: {e}")
            self._status = TrainingStatus.FAILED
            raise
        except KeyboardInterrupt:
            logger.error("Training interrupted.")
            self._status = TrainingStatus.INTERRUPTED
            raise

        logger.info("--- Training completed successfully. ---")
        self._status = TrainingStatus.SUCCEEDED

    def get_topics(self, n_words=10):
        """
        Retrieve the top words for each topic.

        Parameters
        ----------
        n_words : int
            Number of top words to retrieve for each topic.

        Returns
        -------
        list of list of str
            List of topics with each topic represented as a list of top words.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        return [
            [word for word, _ in self.topic_dict[key][:n_words]]
            for key in self.topic_dict
        ]

    def get_topic_word_matrix(self):
        """
        Retrieve the topic-word distribution matrix.

        Returns
        -------
        numpy.ndarray
            Topic-word distribution matrix.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        return self.topic_word_distribution

    def get_topic_document_matrix(self):
        """
        Retrieve the topic-document distribution matrix.

        Returns
        -------
        numpy.ndarray
            Topic-document distribution matrix.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        return self.topic_document_matrix

    def predict(self, documents):
        pass
