from datetime import datetime

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import optuna
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

        self.hparams.update(self.umap_args)
        self.hparams.update(self.kmeans_args)

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

    def _prepare_embeddings(self, dataset):
        """
        Prepares the dataset for clustering.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be used for clustering.
        """

        if dataset.has_embeddings(self.embedding_model_name):
            logger.info(
                f"--- Loading precomputed {EMBEDDING_MODEL_NAME} embeddings ---"
            )
            self.embeddings = dataset.get_embeddings(
                self.embedding_model_name,
                self.embeddings_path,
                self.embeddings_file_path,
            )
            self.dataframe = dataset.dataframe
        else:
            logger.info(f"--- Creating {EMBEDDING_MODEL_NAME} document embeddings ---")
            self.embeddings = self.encode_documents(
                dataset.texts, encoder_model=self.embedding_model_name, use_average=True
            )
            if self.save_embeddings:
                dataset.save_embeddings(
                    self.embeddings,
                    self.embedding_model_name,
                    self.embeddings_path,
                    self.embeddings_file_path,
                )
        self.dataframe = dataset.dataframe

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

        self.n_topics = n_topics

        if self.n_topics <= 0:
            raise ValueError("Number of topics must be greater than 0.")

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

    @staticmethod
    def calculate_aic(n, k, wcss):
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
        return n * np.log(wcss / n) + 2 * k

    @staticmethod
    def calculate_bic(n, k, wcss):
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
        return n * np.log(wcss / n) + k * np.log(n)

    def optimize_hyperparameters(
        self,
        dataset: TMDataset,
        min_topics: int = 2,
        max_topics: int = 20,
        criterion: str = "aic",
        n_trials: int = 100,
        custom_metric=None,
    ):
        """
        Optimize UMAP and K-Means parameters along with the number of topics using AIC, BIC, or a custom metric with Optuna.

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
        assert criterion in [
            "aic",
            "bic",
            "custom",
        ], "Criterion must be either 'aic', 'bic', or 'custom'."
        if criterion == "custom":
            assert (
                custom_metric is not None
            ), "Custom metric must be provided for criterion 'custom'."

        self._prepare_embeddings(dataset)

        def objective(trial):
            # Suggest UMAP parameters
            umap_n_neighbors = trial.suggest_int("umap_n_neighbors", 10, 50)
            umap_n_components = trial.suggest_int("umap_n_components", 5, 50)
            umap_metric = trial.suggest_categorical(
                "umap_metric", ["cosine", "euclidean"]
            )

            # Suggest K-Means parameters
            kmeans_init = trial.suggest_categorical(
                "kmeans_init", ["k-means++", "random"]
            )
            kmeans_n_init = trial.suggest_int("kmeans_n_init", 10, 30)
            kmeans_max_iter = trial.suggest_int("kmeans_max_iter", 100, 1000)

            # Suggest number of topics
            n_topics = trial.suggest_int("n_topics", min_topics, max_topics)

            # Update UMAP and K-Means arguments
            self.umap_args.update(
                {
                    "n_neighbors": umap_n_neighbors,
                    "n_components": umap_n_components,
                    "metric": umap_metric,
                }
            )
            self.kmeans_args.update(
                {
                    "init": kmeans_init,
                    "n_init": kmeans_n_init,
                    "max_iter": kmeans_max_iter,
                }
            )

            self.hparams.update(self.kmeans_args)
            self.hparams.update(self.umap_args)

            # Perform dimensionality reduction
            self.fit(dataset)
            wcss = self.clustering_model.inertia_

            # Calculate AIC, BIC, or custom metric score
            if criterion in ["aic", "bic"]:
                n_samples = self.reduced_embeddings.shape[0]
                if criterion == "aic":
                    score = self.calculate_aic(n_samples, n_topics, wcss)
                else:
                    score = self.calculate_bic(n_samples, n_topics, wcss)
            else:
                # Compute the custom metric score
                topics = self.get_topics()
                score = -custom_metric.score(
                    topics
                )  # Assuming higher metric score is better, negate for minimization

            return score

        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value
        best_n_topics = best_params.pop("n_topics")

        logger.info(
            f"Optimal parameters: {best_params} with {best_n_topics} topics based on {criterion.upper()}."
        )

        return {
            "best_params": best_params,
            "optimal_n_topics": best_n_topics,
            "best_score": best_score,
        }

    def fit_with_optimized_params(
        self,
        dataset: TMDataset,
        min_topics: int = 2,
        max_topics: int = 20,
        criterion: str = "aic",
        n_trials: int = 100,
        custom_metric=None,
    ):
        """
        Optimize hyperparameters and train the model with the optimal parameters using Optuna.

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
        """
        optimal_params = self.optimize_hyperparameters(
            dataset, min_topics, max_topics, criterion, n_trials, custom_metric
        )

        self.umap_args.update(
            {
                "n_neighbors": optimal_params["best_params"]["umap_n_neighbors"],
                "n_components": optimal_params["best_params"]["umap_n_components"],
                "metric": optimal_params["best_params"]["umap_metric"],
            }
        )
        self.kmeans_args.update(
            {
                "init": optimal_params["best_params"]["kmeans_init"],
                "n_init": optimal_params["best_params"]["kmeans_n_init"],
                "max_iter": optimal_params["best_params"]["kmeans_max_iter"],
            }
        )
        self.fit(dataset, n_topics=optimal_params["optimal_n_topics"])
