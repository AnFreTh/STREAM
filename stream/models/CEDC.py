import umap.umap_ as umap
from .abstract_model import BaseModel
from ..utils.encoder import SentenceEncodingMixin
from ..utils.topic_extraction import TopicExtractor
from ..utils.cleaning import clean_topics
from ..data_utils.dataset import TMDataset
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer

data_dir = "../preprocessed_datasets"


class CEDC(BaseModel, SentenceEncodingMixin):
    """
    Class for Clustering-based Embedding-driven Document Clustering (CEDC).

    Inherits from BaseModel and SentenceEncodingMixin.

    Attributes:
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

    Methods
    -------
    get_info()
        Returns a dictionary containing information about the model.
    fit(dataset, n_topics=20)
        Trains the model on the provided dataset and extracts topics.
    predict(texts)
        Predicts topics for new documents.
    get_topics(n_words=10)
        Retrieves the top words for each topic.
    get_topic_word_matrix()
        Retrieves the topic-word distribution matrix.
    get_topic_document_matrix()
        Retrieves the topic-document distribution matrix.
    """

    def __init__(
        self,
        embedding_model_name: str = "paraphrase-MiniLM-L3-v2",
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

        super().__init__(**kwargs)
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

        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path
        self.trained = False
        self.save_embeddings = save_embeddings

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
            "model_name": "CEDC",
            "num_topics": self.n_topics,
            "embedding_model": self.embedding_model_name,
            "umap_args": self.umap_args,
            "kmeans_args": self.gmm_args,
            "trained": self.trained,
        }
        return info

    def _prepare_data(self, dataset):
        """
        Prepares the dataset for clustering.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be used for clustering.
        """

        if dataset.has_embeddings(self.embedding_model_name):

            self.embeddings = dataset.get_embeddings(
                self.embedding_model_name,
                self.embeddings_path,
                self.embeddings_file_path,
            )
            self.dataframe = dataset.dataframe

        else:
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
            print("--- Creating document cluster ---")
            self.GMM = GaussianMixture(
                **self.gmm_args,
            ).fit(self.reduced_embeddings)

            gmm_predictions = self.GMM.predict_proba(self.reduced_embeddings)
            self.soft_labels = pd.DataFrame(gmm_predictions)
            self.labels = self.GMM.predict(self.reduced_embeddings)

        except Exception as e:
            raise ValueError(f"Error in clustering: {e}")

    def _dim_reduction(self):
        """
        Reduces the dimensionality of embeddings using UMAP.

        Raises
        ------
        ValueError
            If an error occurs during dimensionality reduction.
        """
        try:
            print("--- Reducing dimensions ---")
            self.reducer = umap.UMAP(**self.umap_args)
            self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
        except Exception as e:
            raise ValueError(f"Error in dimensionality reduction: {e}")

    def fit(
        self,
        dataset: TMDataset = None,
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

        self.n_topics = n_topics
        self._prepare_data(dataset)
        self._dim_reduction()
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

        print("--- Extract topics ---")
        topics, self.topic_centroids = TE._noun_extractor_haystack(
            self.embeddings,
            n=n_words + 20,
            corpus=expansion_corpus,
            only_nouns=only_nouns,
        )

        self.trained = True

        if clean:
            print("--- Cleaning topics ---")
            cleaned_topics, cleaned_centroids = clean_topics(
                topics, similarity=clean_threshold, embedding_model=self.embedding_model
            )
            topics = cleaned_topics
            self.topic_centroids = cleaned_centroids

        self.topic_word_distribution = self.get_topic_word_matrix(topics)
        self.topic_dict = topics
        self.document_topic_distribution = np.array(self.soft_labels.T)

    def get_topics(self, n_words=10):
        """
        Retrieve the top words for each topic.

        Parameters
        ----------
        n_words : int
            Number of top words to retrieve for each topic.

        Returns
        -------
        list of list of str
            List of topics with each topic represented as a list of top words.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        words_list = []
        new_topics = {}
        for k in range(self.n_topics):
            words = [
                word
                for t in self.topic_dict[k][0:n_words]
                for word in t
                if isinstance(word, str)
            ]
            weights = [
                weight
                for t in self.topic_dict[k][0:n_words]
                for weight in t
                if isinstance(weight, float)
            ]
            weights = [weight / sum(weights) for weight in weights]
            new_topics[k] = list(zip(words, weights))
            words_list.append(words)

        return words_list

    def get_topic_document_matrix(self):
        """
        Retrieves the topic-document matrix if the model is trained.

        Returns
        -------
        ndarray or None
            Topic-document matrix if the model is trained, otherwise None.

        Raises
        ------
        RuntimeError
            If the model is not trained before accessing the topic-document matrix.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        assert (
            self.trained
        ), "Model must be trained before accessing the topic-document matrix."
        # Safely get the topic-document matrix with a default value of None if not found
        return self.output.get("topic-document-matrix", None)

    def get_topic_word_matrix(self, topic_dict):
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

        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        # Extract all unique words and sort them
        all_words = set(word for topic in topic_dict.values() for word, _ in topic)
        sorted_words = sorted(all_words)

        # Create an empty DataFrame with sorted words as rows and topics as columns
        topic_word_matrix = pd.DataFrame(
            index=sorted_words, columns=sorted(topic_dict.keys()), data=0.0
        )

        # Populate the DataFrame with prevalences
        for topic, words in topic_dict.items():
            for word, prevalence in words:
                if word in topic_word_matrix.index:
                    topic_word_matrix.at[word, topic] = prevalence

        return np.array(topic_word_matrix).T

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
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        embeddings = self.encode_documents(
            texts, encoder_model=self.embedding_model_name, use_average=True
        )
        reduced_embeddings = self.reducer.transform(embeddings)
        if proba:
            labels = self.GMM.predict_proba(reduced_embeddings)
        else:
            labels = self.GMM.predict(reduced_embeddings)
        return labels
