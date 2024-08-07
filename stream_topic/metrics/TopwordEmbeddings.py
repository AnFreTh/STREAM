import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from .constants import (
    EMBEDDING_PATH,
    PARAPHRASE_TRANSFORMER_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
)


class TopwordEmbeddings:
    """
    A class to compute and store the embeddings of topwords to use for embedding-based metrics.

    Attributes
    ----------
    word_embedding_model : SentenceTransformer
        SentenceTransformer model to use for word embeddings.
    cache_to_file : bool
        Whether to cache the embeddings to a file.
    emb_filename : str
        Name of the file to save the embeddings to.
    emb_path : str
        Path to save the embeddings to.
    embedding_dict : dict
        Dictionary to store the embeddings of the topwords.
    """

    def __init__(
        self,
        word_embedding_model: SentenceTransformer = SentenceTransformer(
            PARAPHRASE_TRANSFORMER_MODEL
        ),
        cache_to_file: bool = False,
        emb_filename: str = None,
        emb_path: str = EMBEDDING_PATH,
        create_new_file: bool = True,
    ):
        """
        Initialize the TopwordEmbeddings object.

        Parameters
        ----------
        word_embedding_model : SentenceTransformer
            SentenceTransformer model to use for word embeddings.
        cache_to_file : bool, optional
            Whether to cache the embeddings to a file (default is False).
        emb_filename : str, optional
            Name of the file to save the embeddings to (default is None).
        emb_path : str, optional
            Path to save the embeddings to (default is "/embeddings/").
        create_new_file : bool, optional
            Whether to create a new file to save the embeddings to (default is True).
        """
        self.word_embedding_model = word_embedding_model
        self.cache_to_file = cache_to_file
        self.emb_filename = emb_filename
        self.emb_path = emb_path
        self.embedding_dict = {}  # Dictionary to store the embeddings of the topwords

        if self.emb_filename is None:
            self.emb_filename = str(f"Topword_embeddings")

        if create_new_file:
            os.makedirs(self.emb_path, exist_ok=True)

    def _load_embedding_dict_from_disc(self):
        """
        Load the embedding dictionary from the disk.
        """
        try:
            self.embedding_dict = pickle.load(
                open(f"{self.emb_path}{self.emb_filename}.pickle", "rb")
            )
        except FileNotFoundError:
            self.embedding_dict = {}

    def _save_embedding_dict_to_disc(self):
        """
        Save the embedding dictionary to the disk.
        """
        with open(f"{self.emb_path}{self.emb_filename}.pickle", "wb") as handle:
            pickle.dump(self.embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def embed_topwords(
        self,
        topwords: np.ndarray,
        n_topwords_to_use: int = 10,
    ) -> np.ndarray:
        """
        Get the embeddings of the n_topwords topwords.

        Parameters
        ----------
        topwords : np.ndarray
            Array of topwords. Has shape (n_topics, n_words).
        n_topwords_to_use : int, optional
            Number of topwords to use (default is 10).
        Returns
        -------
        np.ndarray
            Array of embeddings of the topwords. Has shape (n_topics, n_topwords_to_use, embedding_dim).
        """
        if type(topwords) is not np.ndarray:
            try:
                topwords = np.array(topwords)
            except Exception as e:
                raise ValueError(f"topwords should be a numpy array.") from e

        if not topwords.ndim == 2:
            topwords = topwords.reshape(-1, 1)

        assert np.issubdtype(
            topwords.dtype, np.str_
        ), "topwords should only contain strings."
        assert (
            topwords.shape[1] >= n_topwords_to_use
        ), "n_topwords_to_use should be less than or equal to the number of words in each topic."

        # Get the top n_topwords words
        topwords = topwords[:, :n_topwords_to_use]
        if self.cache_to_file:
            self._load_embedding_dict_from_disc()

        topword_embeddings = []
        for topic in topwords:
            topic_embeddings = []
            for word in topic:
                if self.embedding_dict and word in self.embedding_dict:
                    topic_embeddings.append(self.embedding_dict[word])
                else:
                    embedding = self.word_embedding_model.encode(word)
                    topic_embeddings.append(embedding)
                    self.embedding_dict[word] = embedding
            topword_embeddings.append(topic_embeddings)

        if self.cache_to_file:
            self._save_embedding_dict_to_disc()
        topword_embeddings = np.array(topword_embeddings)
        topword_embeddings = np.squeeze(topword_embeddings)

        return topword_embeddings
