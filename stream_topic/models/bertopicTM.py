from datetime import datetime

import hdbscan
import numpy as np
from loguru import logger
from sklearn.preprocessing import OneHotEncoder

from ..commons.check_steps import check_dataset_steps
from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.mixins import SentenceEncodingMixin

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "BERTopicTM"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


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

        self.hparams["umap_args"] = self.umap_args
        self.hparams["hdbscan_args"] = self.hdbscan_args

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

    def fit(self, dataset, n_topics=None):
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
            self.dataset, self.embeddings = self.prepare_embeddings(dataset, logger)
            self.dataframe = self.dataset.dataframe
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

            self.beta = tfidf
            self.theta = predictions_one_hot
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

    def suggest_hyperparameters(self, trial):
        """
        Suggests hyperparameters for the model using an Optuna trial.

        This method uses an Optuna trial object to suggest a set of hyperparameters for the model.
        The suggested hyperparameters are stored in the `hparams` dictionary of the model.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The Optuna trial object used for suggesting hyperparameters.
        """
        # Suggest UMAP parameters
        self.hparams["umap_args"]["n_neighbors"] = trial.suggest_int(
            "n_neighbors", 10, 50
        )
        self.hparams["umap_args"]["n_components"] = trial.suggest_int(
            "n_components", 5, 50
        )
        self.hparams["umap_args"]["metric"] = trial.suggest_categorical(
            "metric", ["cosine", "euclidean"]
        )

        # Suggest HDBSCAN parameters
        self.hparams["hdbscan_args"]["min_cluster_size"] = trial.suggest_int(
            "min_cluster_size", 5, 100
        )
        self.hparams["hdbscan_args"]["min_samples"] = trial.suggest_int(
            "min_samples", 1, 100
        )
        self.hparams["hdbscan_args"]["cluster_selection_epsilon"] = trial.suggest_float(
            "cluster_selection_epsilon", 0.0, 1.0
        )

        self.umap_args = self.hparams.get("umap_args")
        self.hdbscan_args = self.hparams.get("hdbscan_args")

    def optimize_and_fit(
        self,
        dataset,
        min_topics=2,
        max_topics=20,
        criterion="aic",
        n_trials=100,
        custom_metric=None,
    ):
        """
        A new method in the child class that calls the parent class's optimize_hyperparameters method.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.
        min_topics : int, optional
            Minimum number of topics to evaluate, by default 2.
        max_topics : int, optional
            Maximum number of topics to evaluate, by default 20.
        criterion : str, optional
            Criterion to use for optimization ('aic', 'bic', or 'custom'), by default 'aic'.
        n_trials : int, optional
            Number of trials for optimization, by default 100.
        custom_metric : object, optional
            Custom metric object with a `score` method for evaluation, by default None.

        Returns
        -------
        dict
            Dictionary containing the best parameters and the optimal number of topics.
        """
        best_params = super().optimize_hyperparameters(
            dataset=dataset,
            min_topics=min_topics,
            max_topics=max_topics,
            criterion=criterion,
            n_trials=n_trials,
            custom_metric=custom_metric,
        )

        return best_params

    def _compute_wcss(self):
        """
        Compute the within-cluster sum of squares (WCSS) for HDBSCAN clusters.

        Returns
        -------
        float
            The WCSS value.
        """
        wcss = 0.0
        labels = self.clustering_model.labels_
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            cluster_points = self.reduced_embeddings[labels == label]
            centroid = cluster_points.mean(axis=0)
            wcss += ((cluster_points - centroid) ** 2).sum()
        return wcss

    def calculate_aic(self, n_topics=None):
        """
        Calculate the AIC for the HDBSCAN model.

        Returns
        -------
        float
            AIC score.
        """
        wcss = self._compute_wcss()
        n = self.reduced_embeddings.shape[0]
        k = len(np.unique(self.clustering_model.labels_))
        return n * np.log(wcss / n) + 2 * k

    def calculate_bic(self, n_topics=None):
        """
        Calculate the BIC for the HDBSCAN model.

        Returns
        -------
        float
            BIC score.
        """
        wcss = self._compute_wcss()
        n = self.reduced_embeddings.shape[0]
        k = len(np.unique(self.clustering_model.labels_))
        return n * np.log(wcss / n) + k * np.log(n)
