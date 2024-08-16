import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
from loguru import logger
from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus

MODEL_NAME = "NMFTM"
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class NMFTM(BaseModel):
    """
    A topic modeling class that uses Non-negative Matrix Factorization (NMF) to cluster text data into topics.

    This class inherits from the BaseModel class and utilizes TF-IDF or Bag-of-Words for vectorization and NMF for dimensionality reduction and clustering.

    Parameters
    ----------
    max_features : int
        Maximum number of features used for vectorization.
    nmf_args : dict
        Arguments for NMF clustering.
    use_tfidf : bool
        If True, use TF-IDF vectorization; if False, use Bag-of-Words.
    tfidf_args : dict
        Arguments for TF-IDF vectorization.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        max_features: int = 5000,
        nmf_args: dict = None,
        use_tfidf: bool = True,
        tfidf_args: dict = None,
        random_state: int = None,
        **kwargs,
    ):
        """
        Initialize the NMF model.

        Parameters
        ----------
        max_features : int, optional
            Maximum number of features used for vectorization, by default 5000.
        nmf_args : dict, optional
            Arguments for NMF clustering, by default None.
        use_tfidf : bool, optional
            If True, use TF-IDF; otherwise, use Bag-of-Words, by default True.
        tfidf_args : dict, optional
            Arguments for TF-IDF vectorization, by default None.
        random_state : int, optional
            Random state for reproducibility, by default None.
        **kwargs
            Additional keyword arguments passed to the superclass.
        """
        super().__init__(use_pretrained_embeddings=False, **kwargs)
        self.save_hyperparameters(ignore=["random_state"])

        self.hparams = {
            "max_features": max_features,
            "nmf_args": nmf_args or {},
            "tfidf_args": tfidf_args
            or {
                "max_df": 0.95,
                "min_df": 2,
                "max_features": max_features,
            },
        }

        if random_state is not None:
            self.hparams["nmf_args"]["random_state"] = random_state

        # Choose vectorizer based on the user's preference
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(**self.hparams["tfidf_args"])
        else:
            self.vectorizer = CountVectorizer(max_features=max_features)

        self._status = TrainingStatus.NOT_STARTED
        self.nmf_model = None
        self.use_tfidf = use_tfidf

    def get_info(self):
        """
        Get information about the model.

        Returns
        -------
        dict
            Dictionary containing model information including model name, vectorization, and clustering arguments, and training status.
        """
        info = {
            "model_name": MODEL_NAME,
            "nmf_args": self.hparams["nmf_args"],
            "vectorizer": "TF-IDF" if self.use_tfidf else "Bag-of-Words",
            "tfidf_args": self.hparams["tfidf_args"],
            "trained_status": self._status.name,
        }
        return info

    def _clustering(self, matrix):
        """
        Applies NMF clustering to the matrix.

        Parameters
        ----------
        matrix : sparse matrix
            The matrix to apply NMF to.

        Raises
        ------
        RuntimeError
            If an error occurs during clustering.
        """
        try:
            logger.info("--- Applying NMF clustering ---")
            self.nmf_model = NMF(
                n_components=self.n_topics,
                **self.hparams["nmf_args"],
            )

            W = self.nmf_model.fit_transform(matrix)  # Document-topic matrix (Theta)
            H = self.nmf_model.components_  # Topic-term matrix (Beta)

            # Assigning attributes
            self.labels = np.argmax(W, axis=1)
            self.theta = W
            self.beta = H

        except Exception as e:
            raise RuntimeError(f"Error in clustering: {e}") from e

    def fit(self, dataset: TMDataset, n_topics: int = 20):
        """
        Trains the NMF topic model on the provided dataset.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.
        n_topics : int, optional
            Number of topics to extract, by default 20.

        Raises
        ------
        RuntimeError
            If the training fails due to an error.
        """
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        self.n_topics = n_topics
        self.dataset = dataset

        self._status = TrainingStatus.RUNNING
        try:
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            matrix = self.vectorizer.fit_transform(self.dataset.texts)
            self._clustering(matrix)

            # Prepare data for visualization
            topic_data = pd.DataFrame(columns=["predictions", "text"])
            for i in range(self.nmf_model.n_components_):
                topic_texts = [
                    self.dataset.texts[j]
                    for j, z in enumerate(self.theta[:, i])
                    if z > 0.1
                ]
                if not topic_texts:
                    continue
                aggregated_texts = " ".join(topic_texts)
                new_row = pd.DataFrame({"predictions": [i], "text": [aggregated_texts]})
                topic_data = pd.concat([topic_data, new_row], ignore_index=True)

            if topic_data.empty:
                raise RuntimeError("No topics were extracted, model training failed.")

            tfidf, count = c_tf_idf(
                topic_data["text"].tolist(), len(self.dataset.texts)
            )
            self.topic_dict = extract_tfidf_topics(tfidf, count, topic_data)

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
        Predict topics for new documents based on their text.

        Parameters
        ----------
        texts : list of str
            List of texts to predict topics for.

        Returns
        -------
        list of int
            List of predicted topic labels.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        matrix = self.vectorizer.transform(texts)
        W = self.nmf_model.transform(matrix)
        return np.argmax(W, axis=1)

    def optimize_and_fit(
        self,
        dataset,
        min_topics=2,
        max_topics=20,
        criterion="recon",
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

    def reconstruction_loss(self):
        """
        Calculate the reconstruction loss (Frobenius norm) for the NMF model.

        Returns
        -------
        float
            Reconstruction loss (Frobenius norm) of the NMF model.
        """
        # Ensure the NMF model has been trained
        if self.nmf_model is None:
            raise ValueError("NMF model has not been trained yet.")

        # Get the original matrix
        original_matrix = self.vectorizer.transform(self.dataset.texts)

        # Reconstruct the matrix using W and H
        reconstructed_matrix = np.dot(self.theta, self.beta)

        # Calculate the Frobenius norm of the difference
        reconstruction_loss = np.linalg.norm(
            original_matrix - reconstructed_matrix, "fro"
        )

        return reconstruction_loss

    def suggest_hyperparameters(self, trial):
        """
        Suggest hyperparameters for the NMF model.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A single trial of the optuna optimization process.

        Returns
        -------
        None
        """

        # Suggest NMF parameters
        # self.hparams["nmf_args"]["alpha_W"] = trial.suggest_float("alpha_W", 0.0, 0.1)
        # self.hparams["nmf_args"]["alpha_H"] = trial.suggest_float("alpha_H", 0.0, 0.1)
        self.hparams["nmf_args"]["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        self.hparams["nmf_args"]["init"] = trial.suggest_categorical(
            "init", ["random", "nndsvd", "nndsvda", "nndsvdar"]
        )
        self.hparams["nmf_args"]["max_iter"] = trial.suggest_int("max_iter", 200, 1000)
        # self.hparams["nmf_args"]["beta_loss"] = trial.suggest_categorical(
        #    "beta_loss", ["frobenius", "kullback-leibler", "itakura-saito"]
        # )
        self.hparams["nmf_args"]["solver"] = trial.suggest_categorical(
            "solver", ["cd", "mu"]
        )

        # if self.hparams["nmf_args"]["beta_loss"] in [
        #    "kullback-leibler",
        #    "itakura-saito",
        # ]:
        #    self.hparams["nmf_args"]["solver"] = "mu"
