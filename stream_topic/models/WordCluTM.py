from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger
from sklearn.mixture import GaussianMixture

from ..commons.check_steps import check_dataset_steps
from ..preprocessor._embedder import BaseEmbedder, GensimBackend
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "WordCluTM"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class WordCluTM(BaseModel):
    """
    A topic modeling class that uses Word2Vec embeddings and K-Means or GMM clustering on vocabulary to form coherent word clusters.
    """

    def __init__(
        self,
        umap_args: dict = None,
        random_state: int = None,
        gmm_args: dict = None,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        **kwargs,
    ):
        """
        Initialize the WordCluTM model.

        Args:
            num_topics (int): Number of topics.
            vector_size (int): Dimensionality of the word vectors.
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker threads to train the Word2Vec model.
            umap_args (dict): Arguments for UMAP dimensionality reduction.
            gmm_args (dict): Arguments for Gaussian Mixture Model (GMM).
            random_state (int): Random seed.
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
        self.n_topics = None

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
            "umap_args": self.umap_args,
            "trained": self._status.name,
        }
        return info

    def train_word2vec(
        self, sentences, epochs, vector_size, window, min_count, workers
    ):
        """
        Train a Word2Vec model on the given sentences.

        Args:
            sentences (list): List of tokenized sentences.
        """
        # Initialize Word2Vec model
        self.word2vec_model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
        )

        # Build the vocabulary from the sentences
        self.word2vec_model.build_vocab(sentences)

        # Train the Word2Vec model
        self.word2vec_model.train(
            sentences, total_examples=len(sentences), epochs=epochs
        )

        # Initialize BaseEmbedder with GensimBackend
        self.base_embedder = BaseEmbedder(
            GensimBackend(self.word2vec_model.wv))

    def _clustering(self):
        """
        Applies GMM clustering to the reduced Word embeddings.

        Raises
        ------
        ValueError
            If an error occurs during clustering.
        """
        assert (
            hasattr(
                self, "reduced_embeddings") and self.reduced_embeddings is not None
        ), "Reduced embeddings must be generated before clustering."

        self.gmm_args["n_components"] = self.n_topics

        try:
            logger.info("--- Creating Word cluster ---")
            self.GMM = GaussianMixture(
                **self.gmm_args,
            ).fit(self.reduced_embeddings)

            gmm_predictions = self.GMM.predict_proba(self.reduced_embeddings)
            self.beta = pd.DataFrame(gmm_predictions)

        except Exception as e:
            raise RuntimeError(f"Error in clustering: {e}") from e

    def fit(
        self,
        dataset: TMDataset = None,
        n_topics: int = 20,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        n_words=10,
        word2vec_epochs=100,
    ):

        self.vector_size = vector_size

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        check_dataset_steps(dataset, logger, MODEL_NAME)

        self.n_topics = n_topics
        if self.n_topics <= 0:
            raise ValueError("Number of topics must be greater than 0.")

        sentences = dataset.get_corpus()
        self._status = TrainingStatus.INITIALIZED

        try:
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            self.train_word2vec(
                sentences=sentences,
                epochs=word2vec_epochs,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers,
            )  # Train Word2Vec model

            logger.info(f"--- Compute word embeddings ---")
            unique_words = list(
                set(word for sentence in sentences for word in sentence)
            )
            word_to_index = {word: i for i, word in enumerate(unique_words)}
            self.embeddings = np.array(
                [
                    (
                        self.word2vec_model.wv[word]
                        if word in self.word2vec_model.wv
                        else np.zeros(vector_size)
                    )
                    for word in unique_words
                ]
            )

            self.reduced_embeddings = self.dim_reduction(logger)
            self._clustering()

            doc_topic_distributions = []
            self.doc_embeddings = []
            logger.info(f"--- Compute doc embeddings ---")
            for doc in sentences:
                # Collect word embeddings for the document
                word_embeddings = [
                    self.word2vec_model.wv[word]
                    for word in doc
                    if word in self.word2vec_model.wv
                ]
                # Compute the mean embedding for the document if there are valid word embeddings
                if word_embeddings:
                    self.doc_embeddings.append(
                        np.mean(word_embeddings, axis=0))
                else:
                    # Append a zero array if no valid word embeddings are found
                    self.doc_embeddings.append(np.zeros(self.vector_size))

            # Replace any NaN values in the final list with zero arrays
            self.doc_embeddings = [
                arr if not np.isnan(arr).any() else np.zeros(arr.shape)
                for arr in self.doc_embeddings
            ]
            if len(self.doc_embeddings) > 0:
                # Reduce the dimensionality of the document embedding
                reduced_doc_embedding = self.reducer.transform(
                    self.doc_embeddings)
                # Predict the topic distribution for the reduced document embedding
                doc_topic_distribution = self.GMM.predict_proba(
                    reduced_doc_embedding)
                # Add the topic distribution to the list
                doc_topic_distributions.append(doc_topic_distribution[0])

            self.theta = pd.DataFrame(doc_topic_distributions)

            # Create topic_dict
            self.topic_dict = {}
            for topic_idx in range(self.beta.shape[1]):
                # Get the indices of the words sorted by their probability of belonging to this topic, in descending order
                sorted_indices = np.argsort(self.beta.iloc[:, topic_idx])[::-1]
                # Get the top n_words for this topic based on the sorted indices
                top_words = [
                    (unique_words[i], self.beta.iloc[i, topic_idx])
                    for i in sorted_indices[:n_words]
                ]
                # Store the top words and their probabilities in the dictionary
                self.topic_dict[topic_idx] = top_words

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
        pass
