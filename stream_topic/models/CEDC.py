from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

from ..commons.check_steps import check_dataset_steps
from ..preprocessor import clean_topics
from ..preprocessor.topic_extraction import TopicExtractor
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.mixins import SentenceEncodingMixin

DATADIR = "../datasets/preprocessed_datasets"
MODEL_NAME = "CEDC"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class CEDC(BaseModel, SentenceEncodingMixin):
    """
    Class for Clustering-based Embedding-driven Document Clustering (CEDC).
    Inherits from BaseModel and SentenceEncodingMixin.


    Parameters
    ----------
    n_topics : int or None
        Number of topics to extract.
    embedding_model_name : str
        Name of the embedding model to use.
    umap_args : dict
        Arguments for UMAP dimensionality reduction.
    gmm_args : dict
        Arguments for Gaussian Mixture Model (GMM) clustering.
    embeddings_path : str
        Path to the folder containing embeddings.
    embeddings_file_path : str
        Path to the file containing embeddings.
    trained : bool
        Flag indicating whether the model has been trained.
    save_embeddings : bool
        Whether to save generated embeddings.

    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        umap_args: dict = None,
        random_state: int = None,
        gmm_args: dict = None,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        **kwargs,
    ):
        """
        Initializes the CEDC model.


        Parameters
        ----------
        embedding_model_name : str, optional
            Name of the embedding model (default is "paraphrase-MiniLM-L3-v2").
        umap_args : dict, optional
            Arguments for UMAP dimensionality reduction.
        random_state : int, optional
            Random state for reproducibility.
        gmm_args : dict, optional
            Arguments for Gaussian Mixture Model (GMM) clustering.
        embeddings_folder_path : str, optional
            Path to the folder containing embeddings.
        embeddings_file_path : str, optional
            Path to the file containing embeddings.
        save_embeddings : bool, optional
            Whether to save generated embeddings.
        **kwargs
            Additional keyword arguments passed to super().__init__().
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

        # Initialize hyperparameters from self.hparams
        self.n_topics = None
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

        self.gmm_args = self.hparams.get(
            "gmm_args",
            gmm_args
            or {
                "n_components": None,
                "covariance_type": "full",
                "tol": 0.001,
                "reg_covar": 0.000001,
                "max_iter": 100,
                "n_init": 1,
                "init_params": "kmeans",
            },
        )

        if random_state is not None:
            self.umap_args["random_state"] = random_state

        self.hparams["umap_args"] = self.umap_args
        self.hparams["gmm_args"] = self.gmm_args

        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path
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
            "num_topics": self.n_topics,
            "embedding_model": self.embedding_model_name,
            "umap_args": self.umap_args,
            "kmeans_args": self.gmm_args,
            "trained": self._status.name,
        }
        return info

    def _clustering(self):
        """
        Applies GMM clustering to the reduced embeddings.


        Raises
        ------
        ValueError
            If an error occurs during clustering.
        """
        assert (
            hasattr(self, "reduced_embeddings") and self.reduced_embeddings is not None
        ), "Reduced embeddings must be generated before clustering."

        self.gmm_args["n_components"] = self.n_topics

        try:
            logger.info("--- Creating document cluster ---")
            self.GMM = GaussianMixture(
                **self.gmm_args,
            ).fit(self.reduced_embeddings)

            gmm_predictions = self.GMM.predict_proba(self.reduced_embeddings)
            self.soft_labels = pd.DataFrame(gmm_predictions)
            self.labels = self.GMM.predict(self.reduced_embeddings)

        except Exception as e:
            raise RuntimeError(f"Error in clustering: {e}") from e

    def fit(
        self,
        dataset: TMDataset,
        n_topics: int = 20,
        only_nouns: bool = False,
        clean: bool = False,
        clean_threshold: float = 0.85,
        expansion_corpus: str = "octis",
        n_words: int = 20,
    ):
        """
        Trains the CEDC model on the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing texts to cluster.
        n_topics : int, optional
            Number of topics to extract (default is 20).
        only_nouns : bool, optional
            Whether to consider only nouns during topic extraction (default is False).
        clean : bool, optional
            Whether to clean topics based on similarity (default is False).
        clean_threshold : float, optional
            Threshold for cleaning topics based on similarity (default is 0.85).
        expansion_corpus : str, optional
            Corpus for expanding topics (default is 'octis').
        n_words : int, optional
            Number of top words to include in each topic (default is 20).


        Returns
        -------
        None
        """

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

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

            assert (
                hasattr(self, "soft_labels") and self.soft_labels is not None
            ), "Clustering must generate labels."

            TE = TopicExtractor(
                dataset=dataset,
                topic_assignments=self.soft_labels,
                n_topics=self.n_topics,
                embedding_model=SentenceTransformer(self.embedding_model_name),
            )

            logger.info("--- Extract topics ---")
            topics, self.topic_centroids = TE._noun_extractor_haystack(
                self.embeddings,
                n=n_words + 20,
                corpus=expansion_corpus,
                only_nouns=only_nouns,
            )

        except Exception as e:
            logger.error(f"Error in training: {e}")
            self._status = TrainingStatus.FAILED
            raise
        except KeyboardInterrupt:
            logger.error("Training interrupted.")
            self._status = TrainingStatus.INTERRUPTED
            raise

        if clean:
            logger.info("--- Cleaning topics ---")
            cleaned_topics, cleaned_centroids = clean_topics(
                topics, similarity=clean_threshold, embedding_model=self.embedding_model
            )
            topics = cleaned_topics
            self.topic_centroids = cleaned_centroids

        logger.info("--- Training completed successfully. ---")
        self._status = TrainingStatus.SUCCEEDED

        self.topic_dict = topics
        self.theta = np.array(self.soft_labels)
        self.beta = self.get_beta()

    def predict(self, texts, proba=True):
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
        if proba:
            labels = self.GMM.predict_proba(reduced_embeddings)
        else:
            labels = self.GMM.predict(reduced_embeddings)
        return labels

    def get_beta(self):
        """
        Constructs a topic-word matrix from the given topic dictionary.

        Parameters
        ----------
        topic_dict : dict
            Dictionary where keys are topic indices and values are lists of (word, prevalence) tuples.

        Returns
        -------
        ndarray
            Topic-word matrix where rows represent topics and columns represent words.


        Notes
        -----
        The topic-word matrix is constructed by assigning prevalences of words in topics.
        Words are sorted alphabetically across all topics.


        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """

        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        assert hasattr(self, "topic_dict"), "Model has no topic_dict."

        # Extract all unique words and sort them
        all_words = set(word for topic in self.topic_dict.values() for word, _ in topic)
        sorted_words = sorted(all_words)

        # Create an empty DataFrame with sorted words as rows and topics as columns
        topic_word_matrix = pd.DataFrame(
            index=sorted_words, columns=sorted(self.topic_dict.keys()), data=0.0
        )

        # Populate the DataFrame with prevalences
        for topic, words in self.topic_dict.items():
            for word, prevalence in words:
                if word in topic_word_matrix.index:
                    topic_word_matrix.at[word, topic] = prevalence
        self.beta = np.array(topic_word_matrix)
        return self.beta

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

        # Suggest GMM parameters
        self.hparams["gmm_args"]["covariance_type"] = trial.suggest_categorical(
            "covariance_type", ["full", "tied", "diag", "spherical"]
        )
        self.hparams["gmm_args"]["tol"] = trial.suggest_float(
            "tol", 1e-4, 1e-1, log=True
        )
        self.hparams["gmm_args"]["reg_covar"] = trial.suggest_float(
            "reg_covar", 1e-6, 1e-3, log=True
        )
        self.hparams["gmm_args"]["max_iter"] = trial.suggest_int("max_iter", 100, 1000)
        self.hparams["gmm_args"]["n_init"] = trial.suggest_int("n_init", 1, 10)
        self.hparams["gmm_args"]["init_params"] = trial.suggest_categorical(
            "init_params", ["kmeans", "random"]
        )

        self.umap_args = self.hparams.get("umap_args")
        self.gmmargs = self.hparams.get("gmm_args")

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
        A new method in the child class that optimizes and fits the model.

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

    def calculate_aic(self, n_topics=None):

        return self.GMM.aic(self.reduced_embeddings)

    def calculate_bic(self, n_topics=None):

        return self.GMM.bic(self.reduced_embeddings)
