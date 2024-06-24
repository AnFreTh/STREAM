import hdbscan
import numpy as np
import umap.umap_ as umap
from loguru import logger
from datetime import datetime
from .base import BaseModel, TrainingStatus
from .mixins import SentenceEncodingMixin
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from ..utils.check_dataset_steps import check_dataset_steps
from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset


time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "BERTopicTM"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class BERTopicTM(BaseModel, SentenceEncodingMixin):
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
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        umap_args: dict = None,
        min_cluster_size: int = None,
        hdbscan_args: dict = None,
        random_state: int = None,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        **kwargs,
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
        super().__init__(use_pretrained_embeddings=True, **kwargs)

        self.save_hyperparameters(
            ignore=[
                "embeddings_file_path",
                "embeddings_folder_path",
                "random_state",
                "save_embeddings",
            ]
        )

        self.embedding_model_name = self.hparams.get(
            "embedding_model_name", embedding_model_name
        )
        self.umap_args = self.hparams.get(
            "umap_args",
            umap_args
            or {
                "n_neighbors": 15,
                "n_components": 15,
                "metric": "cosine",
            },
        )
        if random_state is not None:
            self.umap_args["random_state"] = random_state
        self.min_cluster_size = min_cluster_size
        self.hdbscan_args = self.hparams.get("hdscan_args", hdbscan_args or {})
        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path

        self.save_embeddings = save_embeddings
        self.n_topics = None

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
            "num_topics": self.n_topics,
            "embedding_model": self.embedding_model_name,
            "umap_args": self.umap_args,
            "hdbscan_args": self.hdbscan_args,
            "trained": self._status.name,
        }
        return info

    def _clustering(self):
        """
        Applies K-Means clustering to the reduced embeddings.
        """

        assert (
            hasattr(self, "reduced_embeddings") and self.reduced_embeddings is not None
        ), "Reduced embeddings must be generated before clustering."

        try:
            logger.info("--- Creating document cluster ---")
            self.clustering_model = hdbscan.HDBSCAN(**self.hdbscan_args)
            self.clustering_model.fit(self.reduced_embeddings)
            self.labels = self.clustering_model.labels_
            if self.labels.min() < 0:
                self.labels += 1

        except Exception as e:
            raise RuntimeError(f"Error in clustering: {e}") from e

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

    def fit(self, dataset):
        """
        Trains the BERTOPIC topic model on the provided dataset.

        Applies sentence embedding, UMAP dimensionality reduction, and hdbscan clustering
        to the dataset to identify distinct topics within the text data.

        Parameters:
            dataset: The dataset to train the model on. It should contain the text documents.

        Returns:
            dict: A dictionary containing the identified topics and the topic-word matrix.
        """

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."
        check_dataset_steps(dataset, logger, MODEL_NAME)
        self._status = TrainingStatus.INITIALIZED

        try:
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            self.dataframe, self.embeddings = self.prepare_embeddings(dataset, logger)
            self.reduced_embeddings = self.dim_reduction(logger)

            self._clustering()

            self.dataframe["predictions"] = self.labels
            docs_per_topic = self.dataframe.groupby(
                ["predictions"], as_index=False
            ).agg({"text": " ".join})

            tfidf, count = c_tf_idf(
                docs_per_topic["text"].values, m=len(self.dataframe)
            )

            self.topic_dict = extract_tfidf_topics(tfidf, count, docs_per_topic, n=100)

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
        self.n_topics = len(self.topic_dict)

    def predict(self, texts):
        """
        Predict topics for new documents.

        Parameters
        ----------
        texts : list of str
            List of texts to predict topics for.

        Returns
        -------
        list of int
            List of predicted topic labels.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        embeddings = self.encode_documents(
            texts, encoder_model=self.embedding_model_name, use_average=True
        )
        reduced_embeddings = self.reducer.transform(embeddings)
        labels = self.clustering_model.approximate_predict(reduced_embeddings)
        return labels
