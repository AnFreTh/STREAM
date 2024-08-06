from datetime import datetime

import numpy as np
import optuna
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from ..commons.check_steps import check_dataset_steps
from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.mixins import SentenceEncodingMixin

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "KmeansTM"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class KmeansTM(BaseModel, SentenceEncodingMixin):
    """
    A topic modeling class that uses K-Means clustering on text data.

    This class inherits from the BaseModel class and utilizes sentence embeddings,
    UMAP for dimensionality reduction, and K-Means for clustering text data into topics.

    Parameters
    ----------
    embedding_model_name : str
        Name of the sentence embedding model to use.
    umap_args : dict
        Arguments for UMAP dimensionality reduction.
    kmeans_args : dict
        Arguments for K-Means clustering.
    embeddings_path : str
        Path to the folder containing embeddings.
    embeddings_file_path : str
        Path to the file containing embeddings.
    trained : bool
        Flag indicating whether the model has been trained.
    save_embeddings : bool
        Whether to save generated embeddings.
    n_topics : int or None
        Number of topics to extract.

    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        umap_args: dict = None,
        kmeans_args: dict = None,
        random_state: int = None,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        **kwargs,
    ):
        """
        Initialize the KmeansTM model.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics to extract, by default 20
        embedding_model_name : str, optional
            Name of the sentence embedding model to use, by default "paraphrase-MiniLM-L3-v2"
        umap_args : dict, optional
            Arguments for UMAP dimensionality reduction, by default {}
        kmeans_args : dict, optional
            Arguments for K-Means clustering, by default {}
        random_state : int, optional
            Random state for UMAP, by default None
        embeddings_folder_path : str, optional
            Path to folder to save embeddings, by default None
        embeddings_file_path : str, optional
            Path to specific embeddings file, by default None
        **kwargs
            Additional keyword arguments passed to the superclass.
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
        self.kmeans_args = self.hparams.get("kmeans_args", kmeans_args or {})

        self.hparams["umap_args"] = self.umap_args
        self.hparams["kmeans_args"] = self.kmeans_args

        if random_state is not None:
            self.umap_args["random_state"] = random_state

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
            "kmeans_args": self.kmeans_args,
            "trained": self._status.name,
        }
        return info

    def _clustering(self):
        """
        Applies K-Means clustering to the reduced embeddings.

        Raises
        ------
        ValueError
            If an error occurs during clustering.
        """
        assert (
            hasattr(self, "reduced_embeddings") and self.reduced_embeddings is not None
        ), "Reduced embeddings must be generated before clustering."

        self.kmeans_args = self.hparams.get("kmeans_args", self.kmeans_args)
        try:
            logger.info("--- Creating document cluster ---")
            self.clustering_model = KMeans(n_clusters=self.n_topics, **self.kmeans_args)
            self.clustering_model.fit(self.reduced_embeddings)
            self.labels = self.clustering_model.labels_

            labels = np.array(self.labels)
            self.topic_centroids = []

            for label in np.unique(labels):
                label_embeddings = self.embeddings[labels == label]
                mean_embedding = np.mean(label_embeddings, axis=0)
                self.topic_centroids.append(mean_embedding)

        except Exception as e:
            raise RuntimeError(f"Error in clustering: {e}") from e

    def fit(
        self,
        dataset: TMDataset = None,
        n_topics: int = 20,
    ):
        """
        Trains the K-Means topic model on the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to train the model on.
        n_topics : int, optional
            Number of topics to extract, by default 20

        Raises
        ------
        AssertionError
            If the dataset is not an instance of TMDataset.
        """

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        check_dataset_steps(dataset, logger, MODEL_NAME)
        self.dataset = dataset

        self.n_topics = n_topics

        if self.n_topics <= 0:
            raise ValueError("Number of topics must be greater than 0.")

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
        labels = self.clustering_model.predict(reduced_embeddings)
        return labels

    def calculate_aic(self, n_topics=None):
        """
        Calculate the AIC for a given model.

        Parameters
        ----------
        n : int
            Number of samples.
        k : int
            Number of clusters.
        wcss : float
            Within-cluster sum of squares.

        Returns
        -------
        float
            AIC score.
        """
        wcss = self.clustering_model.inertia_
        n = self.reduced_embeddings.shape[0]
        return n * np.log(wcss / n) + 2 * n_topics

    def calculate_bic(self, n_topics=None):
        """
        Calculate the BIC for a given model.

        Parameters
        ----------
        n : int
            Number of samples.
        k : int
            Number of clusters.
        wcss : float
            Within-cluster sum of squares.

        Returns
        -------
        float
            BIC score.
        """
        wcss = self.clustering_model.inertia_
        n = self.reduced_embeddings.shape[0]
        return n * np.log(wcss / n) + n_topics * np.log(n)

    def suggest_hyperparameters(self, trial):
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

        # Suggest K-Means parameters
        self.hparams["kmeans_args"]["init"] = trial.suggest_categorical(
            "init", ["k-means++", "random"]
        )
        self.hparams["kmeans_args"]["n_init"] = trial.suggest_int("n_init", 10, 30)
        self.hparams["kmeans_args"]["max_iter"] = trial.suggest_int(
            "max_iter", 100, 1000
        )

        self.umap_args = self.hparams.get("umap_args")
        self.kmeans_args = self.hparams.get("kmeans_args")

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
