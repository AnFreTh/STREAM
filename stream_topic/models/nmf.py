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
import pandas as pd

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
        self.stopwords_path = kwargs.get("stopwords_path", None)

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

    def _fit_transform_cached(self):
        """Build NMF's input matrix over the SHARED benchmark vocabulary.

        NMF factorizes TF-IDF (use_tfidf, the canonical NMF setup) computed over
        the SHARED benchmark BOW (dataset.get_bow(), min_df/max_df set during
        preprocessing = 25 / 0.7). This makes NMF's vocabulary IDENTICAL to every
        other model's and fixes its perplexity (previously NMF re-vectorized
        dataset.texts with its own min_df=2/max_df=0.95/max_features=5000 -> a
        different, larger vocab that was incomparable and nulled perplexity).
        get_bow() caches on the dataset (computed once, reused across the 5 seeds;
        no RNG consumed, so NMF's init RNG is unperturbed). The shared vocabulary
        is stored on self.feature_names for topic extraction.
        """
        from sklearn.feature_extraction.text import TfidfTransformer

        bow, vocab = self.dataset.get_bow()
        self.feature_names = vocab
        if self.use_tfidf:
            # L2-normalized TF-IDF (sklearn defaults) over the shared count matrix
            # == what TfidfVectorizer would produce on this vocabulary.
            return TfidfTransformer().fit_transform(bow)
        return bow

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
                max_iter=self.hparams["nmf_args"].get("max_iter", 200),
                **{k: v for k, v in self.hparams["nmf_args"].items() if k != "max_iter"},
            )

            W = self.nmf_model.fit_transform(matrix)  # Document-topic matrix (Theta)
            H = self.nmf_model.components_  # Topic-term matrix (Beta)

            # Assigning attributes
            self.labels = np.argmax(W, axis=1)
            self.theta = W
            self.beta = H

            # Build the topic dictionary directly from the NMF factor matrix H
            # (canonical NMF topics: the top-weighted words of each component),
            # rather than via c-TF-IDF of strongly-assigned documents. This
            # guarantees exactly K topics, each aligned with its beta row, and
            # never drops a topic that happens to have no document with
            # theta > threshold (the previous behaviour returned < K topics and
            # misaligned topics vs beta).
            # Shared benchmark vocabulary (set in _fit_transform_cached from
            # dataset.get_bow()); falls back to the vectorizer only on the
            # explicit user-supplied path.
            feature_names = getattr(self, "feature_names", None)
            if feature_names is None:
                feature_names = self.vectorizer.get_feature_names_out()
            n_top = 100
            self.topic_dict = {}
            for k in range(H.shape[0]):
                top_idx = np.argsort(H[k])[::-1][:n_top]
                self.topic_dict[k] = [
                    (feature_names[j], float(H[k][j])) for j in top_idx
                ]

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
        if self.stopwords_path is not None:
            # stopwords = pd.read_csv(self.stopwords_path, names=['w'], sep='\t', encoding='UTF-8')
            with open(self.stopwords_path, 'r', encoding='UTF-8') as f:
                stop_words = [line.strip() for line in f]
                stopwords = pd.DataFrame({'w': stop_words})
            stopwords_list = set(stopwords['w'])
            try:
                logger.info(f"--- Training {MODEL_NAME} topic model ---")
                matrix = self._fit_transform_cached()
                self._clustering(matrix)  # builds self.topic_dict from beta

            except Exception as e:
                logger.error(f"Error in training: {e}")
                self._status = TrainingStatus.FAILED
                raise
            except KeyboardInterrupt:
                logger.error("Training interrupted.")
                self._status = TrainingStatus.INTERRUPTED
                raise
        else:
            try:
                logger.info(f"--- Training {MODEL_NAME} topic model ---")
                matrix = self._fit_transform_cached()
                self._clustering(matrix)  # builds self.topic_dict from beta

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
        timeout=None,
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
            timeout=timeout,
        )

        return best_params

    def reconstruction_loss(self):
        """
        Calculate the reconstruction loss (Frobenius norm) for the NMF model.
        Uses sklearn's built-in reconstruction_err_ when available.

        Returns
        -------
        float
            Reconstruction loss of the NMF model.
        """
        if self.nmf_model is None:
            raise ValueError("NMF model has not been trained yet.")

        # Use sklearn's built-in attribute (avoids dense matrix explosion)
        if hasattr(self.nmf_model, 'reconstruction_err_'):
            return self.nmf_model.reconstruction_err_

        # Fallback: compute on sparse matrices
        original_matrix = self.vectorizer.transform(self.dataset.texts)
        reconstructed = csr_matrix(np.dot(self.theta, self.beta))
        diff = original_matrix - reconstructed
        return np.sqrt(diff.multiply(diff).sum())

    def suggest_hyperparameters(self, trial):
        self.hparams["nmf_args"]["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        self.hparams["nmf_args"]["init"] = trial.suggest_categorical(
            "init", ["nndsvda", "nndsvdar", "random"]
        )
        self.hparams["nmf_args"]["max_iter"] = trial.suggest_int("max_iter", 100, 400)
        self.hparams["nmf_args"]["solver"] = trial.suggest_categorical(
            "solver", ["cd", "mu"]
        )
        self.hparams["nmf_args"]["tol"] = trial.suggest_float("tol", 1e-5, 1e-2, log=True)
