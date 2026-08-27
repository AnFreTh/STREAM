"""KMeans topic model with PCA dimensionality reduction (deterministic baseline)."""

from datetime import datetime

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

from ..commons.check_steps import check_dataset_steps
from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.mixins import SentenceEncodingMixin

MODEL_NAME = "KmeansTM_PCA"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"


class KmeansTM_PCA(BaseModel, SentenceEncodingMixin):
    """
    KMeans topic model using PCA for dimensionality reduction.

    Identical to KmeansTM but replaces UMAP with PCA, providing a
    deterministic, faster baseline. Useful for ablation studies
    comparing nonlinear (UMAP) vs linear (PCA) dim reduction.

    Parameters
    ----------
    embedding_model_name : str
        Sentence embedding model name.
    n_components : int
        Number of PCA components.
    kmeans_args : dict
        Arguments for KMeans clustering.
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        n_components: int = 15,
        kmeans_args: dict = None,
        random_state: int = 42,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        **kwargs,
    ):
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
        self.n_components = self.hparams.get("n_components", n_components)
        self.kmeans_args = self.hparams.get("kmeans_args", kmeans_args or {})
        self.hparams["kmeans_args"] = self.kmeans_args

        # Keep the run seed so BOTH stochastic stages (PCA randomized-SVD and
        # KMeans) are seeded -- otherwise PCA's svd_solver='auto' dispatches to an
        # UNSEEDED randomized SVD, making the "deterministic baseline" non-
        # reproducible run-to-run.
        self.random_state = random_state
        # Fixed random_state → deterministic clustering (paper claim)
        self.kmeans_args.setdefault("random_state", random_state)
        # n_init=10 (vs modern sklearn's 'auto' -> 1): stronger/stabler baseline,
        # and matches the n_init=10 the PCA HPO path already pins.
        self.kmeans_args.setdefault("n_init", 10)

        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path
        self.save_embeddings = save_embeddings
        self.n_topics = None
        self._status = TrainingStatus.NOT_STARTED

    def get_info(self):
        return {
            "model_name": MODEL_NAME,
            "num_topics": self.n_topics,
            "embedding_model": self.embedding_model_name,
            "n_components": self.n_components,
            "kmeans_args": self.kmeans_args,
            "trained": self._status.name,
        }

    def _dim_reduction(self):
        """PCA dimensionality reduction."""
        logger.info("--- Reducing dimensions with PCA ---")
        # Resync from hparams: the post-HPO refit writes the BEST n_components
        # into hparams, but this method reads the attribute, which otherwise
        # still holds the last trial's value.
        self.n_components = self.hparams.get("n_components", self.n_components)
        n_comp = min(self.n_components, self.embeddings.shape[1], self.embeddings.shape[0])
        # Seed PCA: with svd_solver='auto' sklearn picks randomized SVD for these
        # shapes, which is non-deterministic unless random_state is set.
        self.reducer = PCA(n_components=n_comp, random_state=self.random_state)
        self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)

    def _clustering(self):
        """KMeans clustering on reduced embeddings."""
        self.kmeans_args = self.hparams.get("kmeans_args", self.kmeans_args)
        logger.info("--- Creating document cluster ---")
        self.clustering_model = KMeans(n_clusters=self.n_topics, **self.kmeans_args)
        self.clustering_model.fit(self.reduced_embeddings)
        self.labels = self.clustering_model.labels_
        # topic_centroids is only consumed by stream_topic/visuals; computed
        # lazily via the property below instead of eagerly on every fit/HPO trial.

    @property
    def topic_centroids(self):
        """Mean full-dimensional embedding per cluster (lazy; used by visuals only)."""
        if getattr(self, "_topic_centroids", None) is None:
            labels = np.array(self.labels)
            self._topic_centroids = [
                np.mean(self.embeddings[labels == label], axis=0)
                for label in np.unique(labels)
            ]
        return self._topic_centroids

    def fit(self, dataset: TMDataset = None, n_topics: int = 20):
        assert isinstance(dataset, TMDataset), "Dataset must be a TMDataset instance."
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
            self._dim_reduction()
            self._clustering()

            self.dataframe["predictions"] = self.labels
            docs_per_topic = self.dataframe.groupby(
                ["predictions"], as_index=False
            ).agg({"text": " ".join})

            tfidf, count = c_tf_idf(
                docs_per_topic["text"].values, m=len(self.dataframe)
            )
            self.topic_dict = extract_tfidf_topics(tfidf, count, docs_per_topic, n=100)

            one_hot_encoder = OneHotEncoder(sparse_output=False)
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
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        embeddings = self.encode_documents(
            texts, encoder_model=self.embedding_model_name, use_average=True
        )
        reduced = self.reducer.transform(embeddings)
        return self.clustering_model.predict(reduced)

    def calculate_aic(self, n_topics=None):
        wcss = self.clustering_model.inertia_
        n = self.reduced_embeddings.shape[0]
        return n * np.log(wcss / n) + 2 * n_topics

    def calculate_bic(self, n_topics=None):
        wcss = self.clustering_model.inertia_
        n = self.reduced_embeddings.shape[0]
        return n * np.log(wcss / n) + n_topics * np.log(n)

    def suggest_hyperparameters(self, trial):
        self.hparams["n_components"] = trial.suggest_int("n_components", 5, 50)
        self.n_components = self.hparams["n_components"]

        self.hparams["kmeans_args"]["max_iter"] = trial.suggest_int(
            "max_iter", 100, 1000
        )
        # init and n_init are fixed to keep the model deterministic given a
        # fixed random_state. Tuning them would reintroduce stochasticity.
        self.hparams["kmeans_args"].setdefault("init", "k-means++")
        self.hparams["kmeans_args"].setdefault("n_init", 10)
        self.hparams["kmeans_args"].setdefault("random_state", 42)
        self.kmeans_args = self.hparams.get("kmeans_args")

    def optimize_and_fit(
        self,
        dataset,
        min_topics=2,
        max_topics=20,
        criterion="aic",
        n_trials=100,
        custom_metric=None,
        timeout=None,
    ):
        best_params = super().optimize_hyperparameters(
            dataset=dataset,
            min_topics=min_topics,
            max_topics=max_topics,
            criterion=criterion,
            n_trials=n_trials,
            custom_metric=custom_metric,
            timeout=timeout,
        )
        return best_params
