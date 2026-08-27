from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from loguru import logger
from nltk.tokenize import word_tokenize

from ..commons.check_steps import check_dataset_steps
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
import jieba
import re

MODEL_NAME = "LDA"
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class LDA(BaseModel):

    def __init__(self, vectorizer=None, random_state=None, **kwargs):
        """
        Initialize the LDA model.

        Parameters
        ----------
        vectorizer : CountVectorizer or None, optional
            A scikit-learn CountVectorizer for text preprocessing.
        random_state : int or None, optional
            Seed for random number generation.
        """
        super().__init__(use_pretrained_embeddings=True, **kwargs)
        self.save_hyperparameters(ignore=["vectorizer"])

        self._status = TrainingStatus.NOT_STARTED
        self.n_topics = None
        self.vectorizer = vectorizer
        self.random_state = random_state
        self.doc_term_matrix = None
        self.feature_names = None

    def get_info(self):
        """
        Get information about the LDA model.

        Returns
        -------
        info : dict
            Dictionary containing model information.
        """
        info = {
            "model_name": MODEL_NAME,
            "num_topics": self.n_topics,
            "trained": self._status.name,
        }
        return info

    def _assert_and_tokenize(self, dataset):
        """
        Ensure that the 'tokens' column exists and tokenize the entries if needed.

        Parameters
        ----------
        dataset : TMDataset
            The dataset containing the 'tokens' column.

        Raises
        ------
        ValueError
            If the 'tokens' column does not exist in the dataframe.
        """
        # Ensure the 'tokens' column exists
        if "tokens" not in dataset.dataframe.columns:
            raise ValueError(
                f"Column 'tokens' does not exist in the dataframe.")

        # Define a helper function to check if an entry is tokenized
        def is_tokenized(entry):
            return isinstance(entry, list) and all(
                isinstance(token, str) for token in entry
            )
        def is_chinese(text):
            pattern = re.compile(r'[\u4e00-\u9fff]')
            # Check if at least one Chinese character exists in the text
            return bool(pattern.search(text))
        def tokenize(entry):
            if not is_tokenized(entry):
                if isinstance(entry, str):
                    if is_chinese(entry):  # Check if the text is Chinese
                        return list(jieba.cut(entry))  # Use jieba for Chinese tokenization
                    else:
                        return word_tokenize(entry)  # Use nltk for non-Chinese tokenization
            return entry
        # Tokenize entries that are not tokenized
        dataset.dataframe["tokens"] = dataset.dataframe["tokens"].apply(tokenize)
        # delete space after re-split
        dataset.dataframe["tokens"] = dataset.dataframe["tokens"].apply(lambda tokens: [word for word in tokens if word != ' '])

        return dataset

    def _prepare_documents(self, dataset):
        """
        Prepare the documents for LDA training.

        Parameters
        ----------
        dataset : TMDataset
            The dataset containing the documents to be prepared.
        """

        logger.info(f"--- Preparing the documents for {MODEL_NAME} ---")

        # Use the SHARED benchmark BOW (the min_df/max_df set during preprocessing,
        # 25 / 0.7 in the benchmark) so LDA factorizes the IDENTICAL vocabulary as
        # every other model, and its perplexity (dataset.bow vs LDA's beta) is
        # dimensionally consistent. Previously LDA re-vectorized dataset.texts with
        # its own min_df=2/max_df=0.95 -> a larger, different vocab that made it
        # incomparable and nulled its perplexity. get_bow() caches on the dataset,
        # so this is computed once and reused across the 5 seeds (no RNG consumed,
        # so it does not perturb the RNG the LDA fit draws from).
        if self.vectorizer is not None:
            # Explicit user-supplied vectorizer path (not used in the benchmark).
            documents = dataset.dataframe["text"].tolist()
            self.doc_term_matrix = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            self.doc_term_matrix, self.feature_names = dataset.get_bow()

    def fit(self, dataset: TMDataset = None, n_topics: int = 20, language: str = "en", **lda_params):
        """
        Fit the LDA model to the dataset.

        Parameters
        ----------
        dataset : TMDataset, optional
            The dataset to fit the model to. Must be an instance of TMDataset.
        n_topics : int, optional
            The number of topics to extract (default is 20).
        **lda_params : dict, optional
            Additional parameters to pass to the Gensim LdaModel.

        Raises
        ------
        AssertionError
            If the dataset is not an instance of TMDataset.
        RuntimeError
            If there is an error during training.
        """
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        if language == 'chinese':
            check_dataset_steps(dataset, logger, MODEL_NAME, language='chinese')
        else:
            check_dataset_steps(dataset, logger, MODEL_NAME)
        self.dataset = dataset

        self.n_topics = n_topics

        try:
            self._status = TrainingStatus.INITIALIZED
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            if self.doc_term_matrix is None:
                self._prepare_documents(dataset)
            
            lda_params = {
                key: value
                for key, value in {**self.hparams, **lda_params}.items()
                if key not in ["n_topics", "vectorizer"]
            }
            
            # Set default parameters if not provided
            lda_params.setdefault('random_state', self.random_state)
            lda_params.setdefault('max_iter', 50)  # was 10; batch VI needs more passes to converge
            lda_params.setdefault('learning_method', 'batch')
            # NOTE: sklearn's parallel batch E-step (n_jobs=-1) is theoretically
            # only algebraically equivalent to serial, but empirically it drifts
            # ~70% at theta / ~38% at components on real data over 10 iterations
            # because reassociation errors compound. NOT enabled here -- it is a
            # BEHAVIOR-CHANGING optimization, not fp-equivalent. Left as a
            # documented option for future rerun-with-known-diff experiments.
            
            self.model = LatentDirichletAllocation(
                n_components=n_topics, 
                **lda_params
            )
            self.model.fit(self.doc_term_matrix)
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

        self.theta = self.get_theta()
        self.labels = np.array(np.argmax(self.theta, axis=1))

        self.topic_dict = self._get_topic_word_dict()

    def optimize_and_fit(
        self,
        dataset,
        metric=None,
        min_topics=2,
        max_topics=20,
        criterion="aic",
        n_trials=100,
        timeout=None,
    ):
        """
        Optimize hyperparameters and fit the LDA model.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.
        metric : object, optional
            Custom metric with a score() method. Used when criterion='custom'.
        min_topics : int, optional
            Minimum number of topics, by default 2.
        max_topics : int, optional
            Maximum number of topics, by default 20.
        criterion : str, optional
            'aic', 'bic', or 'custom', by default 'aic'.
        n_trials : int, optional
            Number of HPO trials, by default 100.
        timeout : int, optional
            HPO timeout in seconds, by default None.
        """
        if criterion == "custom" and metric is None:
            raise ValueError("metric must be provided when criterion='custom'")

        best_params = super().optimize_hyperparameters(
            dataset=dataset,
            min_topics=min_topics,
            max_topics=max_topics,
            criterion=criterion if metric is None else "custom",
            n_trials=n_trials,
            custom_metric=metric,
            timeout=timeout,
        )

        return best_params

    def predict(self, dataset):
        pass

    def get_theta(self):
        """
        Get the topic distribution for each document.

        Returns
        -------
        topic_document_matrix : pd.DataFrame
            DataFrame where each row corresponds to a document and each column to a topic,
            with the values representing the topic probabilities for each document.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet or failed.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")

        # transform() over the fixed fitted components_ and DTM is deterministic
        # (sklearn's E-step uses a fixed np.ones init, random_state unused), so the
        # result is invariant across calls. It is invoked 2-3x per run (fit, then
        # in bench_common for perplexity and NMI/Purity), each a full variational
        # E-step; memoize it.
        if getattr(self, "_theta_cache", None) is not None:
            return self._theta_cache

        # Get document-topic distribution
        doc_topic_dist = self.model.transform(self.doc_term_matrix)

        # Convert to DataFrame with proper column names
        columns = [f"topic_{i}" for i in range(self.n_topics)]
        self._theta_cache = pd.DataFrame(doc_topic_dist, columns=columns)
        return self._theta_cache



    def get_beta(self):
        """
        Get the word distribution for each topic.

        Returns
        -------
        beta_matrix : np.ndarray
            Topic-word distribution matrix.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet or failed.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")

        # Get topic-word distribution (components_)
        # Shape: (n_topics, n_features)
        self.beta = self.model.components_.T  # Transpose to get (n_features, n_topics)
        return self.beta

    def _get_topic_word_dict(self, num_words=100):
        """
        Get the topic-word dictionary for the LDA model.

        Parameters
        ----------
        num_words : int, optional
            The number of top words to include for each topic (default is 100).

        Returns
        -------
        topic_word_dict : dict
            Dictionary where keys are topic ids and values are lists of tuples (word, probability).
        """
        topic_word_dict = {}
        
        # Get topic-word distribution
        topic_word_dist = self.model.components_
        
        for topic_id in range(self.n_topics):
            # Get top words for this topic
            top_word_indices = np.argsort(topic_word_dist[topic_id])[::-1][:num_words]
            topic_word_dict[topic_id] = [
                (self.feature_names[word_idx], topic_word_dist[topic_id][word_idx])
                for word_idx in top_word_indices
            ]

        return topic_word_dict

    def calculate_aic(self, n_topics=None):
        """AIC using sklearn LDA's log-likelihood score."""
        log_likelihood = self.model.score(self.doc_term_matrix)
        n_params = n_topics * self.doc_term_matrix.shape[1]  # K * V
        return -2 * log_likelihood + 2 * n_params

    def calculate_bic(self, n_topics=None):
        """BIC using sklearn LDA's log-likelihood score."""
        log_likelihood = self.model.score(self.doc_term_matrix)
        n_samples = self.doc_term_matrix.shape[0]
        n_params = n_topics * self.doc_term_matrix.shape[1]
        return -2 * log_likelihood + n_params * np.log(n_samples)

    def suggest_hyperparameters(self, trial):
        # Suggest LDA-specific hyperparameters for scikit-learn LDA
        self.hparams["doc_topic_prior"] = trial.suggest_float("doc_topic_prior", 0.01, 1.0)
        self.hparams["topic_word_prior"] = trial.suggest_float("topic_word_prior", 0.01, 1.0)
        self.hparams["learning_decay"] = trial.suggest_float("learning_decay", 0.5, 1.0)
        self.hparams["max_iter"] = trial.suggest_int("max_iter", 5, 50)
