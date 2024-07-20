from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger
from sklearn.mixture import GaussianMixture

from ..preprocessor._embedder import BaseEmbedder, GensimBackend
from ..utils.check_dataset_steps import check_dataset_steps
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "WordCluTM"
logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


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
        Applies GMM clustering to the reduced embeddings.

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
            logger.info("--- Creating document cluster ---")
            self.GMM = GaussianMixture(
                **self.gmm_args,
            ).fit(self.reduced_embeddings)

            gmm_predictions = self.GMM.predict_proba(self.reduced_embeddings)
            self.theta = pd.DataFrame(gmm_predictions)
            self.labels = self.GMM.predict(self.reduced_embeddings)

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
            word_embeddings = np.array(
                [
                    (
                        self.word2vec_model.wv[word]
                        if word in self.word2vec_model.wv
                        else np.zeros(vector_size)
                    )
                    for word in unique_words
                ]
            )

            logger.info(f"--- Compute document embeddings ---")
            # Initialize an empty list to hold document embeddings
            self.embeddings = []

            # Iterate over each document to compute its embedding
            for doc in sentences:
                doc_embedding = np.mean(
                    [
                        word_embeddings[word_to_index[word]]
                        for word in doc
                        if word in word_to_index
                    ],
                    axis=0,
                )
                if doc_embedding.shape == (vector_size,):
                    self.embeddings.append(doc_embedding)

            # Convert the list of document embeddings to a numpy array
            self.embeddings = np.array(self.embeddings)
            # self.embeddings = self.embeddings[~np.isnan(self.embeddings).any(axis=1)]

            self.reduced_embeddings = self.dim_reduction(logger)
            self._clustering()

            self.topic_dict = {}

            # Iterate over each cluster
            for cluster_idx in range(self.theta.shape[1]):
                # Get the indices of the words sorted by their probability of belonging to this cluster, in descending order
                sorted_indices = np.argsort(self.theta[:, cluster_idx])[::-1]

                # Get the top n_words for this cluster based on the sorted indices
                top_words = [
                    (unique_words[i], self.theta[i, cluster_idx])
                    for i in sorted_indices[:n_words]
                ]

                # Store the top words and their probabilities in the dictionary
                self.topic_dict[cluster_idx] = top_words

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

    def get_theta(self):
        return self.beta

    def get_beta(self, dataset):
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        assert hasattr(self, "topic_dict"), "Model has no topic_dict."
        corpus = dataset.get_corpus()
        n_documents = len(corpus)
        document_scores_per_label = {label: []
                                     for label in range(len(self.topic_dict))}
        # Convert top_words_per_cluster to a more easily searchable structure
        word_scores_per_label = {}
        for label, words_scores in self.topic_dict.items():
            for word, score in words_scores:
                if word not in word_scores_per_label:
                    word_scores_per_label[word] = {}
                word_scores_per_label[word][label] = score
        # Iterate over each document
        for doc in corpus:
            # Initialize a score accumulator for each label for the current document
            doc_scores = {label: [] for label in range(len(self.topic_dict))}
            # Iterate over each word in the document
            for word in doc:
                if word in word_scores_per_label:
                    # If the word has scores for any label, add those scores to the accumulator
                    for label, score in word_scores_per_label[word].items():
                        doc_scores[label].append(score)
            # Average the scores for each label and store them
            for label in doc_scores:
                if doc_scores[label]:  # Check if there are any scores to average
                    document_scores_per_label[label].append(
                        np.mean(doc_scores[label]))
                else:
                    # If no scores for this label, you might want to set a default value
                    document_scores_per_label[label].append(0)

        # Initialize the final array with zeros
        final_scores = np.zeros((self.n_topics, n_documents))
        # Populate the array with the computed scores
        for label, scores in document_scores_per_label.items():
            # Ensure there are as many scores as there are documents
            assert (
                len(scores) == n_documents
            ), "The number of scores must match the number of documents"
            # Assign the scores for this label (topic) to the corresponding row in the final array
            final_scores[label, :] = scores

        return final_scores

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
        unique_words = list(
            set(word for sentence in texts for word in sentence))
        word_to_index = {word: i for i, word in enumerate(unique_words)}
        word_embeddings = np.array(
            [
                (
                    self.word2vec_model.wv[word]
                    if word in self.word2vec_model.wv
                    else np.zeros(self.vector_size)
                )
                for word in unique_words
            ]
        )

        logger.info(f"--- Compute document embeddings ---")
        # Initialize an empty list to hold document embeddings
        self.doc_embeddings = []

        # Iterate over each document to compute its embedding
        for doc in texts:
            doc_embedding = np.mean(
                [
                    word_embeddings[word_to_index[word]]
                    for word in doc
                    if word in word_to_index
                ],
                axis=0,
            )
            self.doc_embeddings.append(doc_embedding)

        # Convert the list of document embeddings to a numpy array
        embeddings = np.array(self.doc_embeddings)

        reduced_embeddings = self.dim_reduction(logger)

        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")

        reduced_embeddings = self.reducer.transform(embeddings)
        if proba:
            labels = self.GMM.predict_proba(reduced_embeddings)
        else:
            labels = self.GMM.predict(reduced_embeddings)
        return labels
