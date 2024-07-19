
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from .constants import SENTENCE_TRANSFORMER_MODEL


class TopwordEmbeddings:
    """
    A class to compute and store the embeddings of topwords to use for embedding-based metrics.
    """


    def __init__(
            self,
            word_embedding_model: SentenceTransformer = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL),
            emb_filename:str = None,
            emb_path: str ="Embeddings/"
    ):
        """
        Initialize the TopwordEmbeddings object.

        Parameters
        ----------
        word_embedding_model : SentenceTransformer
            SentenceTransformer model to use for word embeddings.
        emb_filename : str, optional
            Name of the file to save the embeddings to (default is None).
        emb_path : str, optional
            Path to save the embeddings to (default is "Embeddings/").
        """
        self.word_embedding_model = word_embedding_model
        self.emb_filename = emb_filename
        self.emb_path = emb_path
        self.embedding_dict = None # Dictionary to store the embeddings of the topwords

        if self.emb_filename is None:
            self.emb_filename = str(f"Topword_embeddings_{word_embedding_model}")


    def _load_embedding_dict_from_disc(self):
        """
        Load the embedding dictionary from the disk.
        """
        try:
            self.embedding_dict = pickle.load(open(f"{self.emb_path}{self.emb_filename}.pickle", "rb"))
        except FileNotFoundError:
            self.embedding_dict = None

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
            cache_embeddings: bool = True
    ) -> np.ndarray:
        """
        Get the embeddings of the n_topwords topwords.

        Parameters
        ----------
        topwords : np.ndarray
            Array of topwords. Has shape (n_topics, n_words).
        n_topwords_to_use : int, optional
            Number of topwords to use (default is 10).
        cache_embeddings : bool, optional
            Whether to cache the embeddings, i.e whether to update the embedding dictionary and save the result to disk (default is True).

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

        assert topwords.ndim == 2, "topwords should be a 2D array."
        assert np.issubdtype(topwords.dtype, np.str_), "topwords should only contain strings."
        assert topwords.shape[1] >= n_topwords_to_use, "n_topwords_to_use should be less than or equal to the number of words in each topic."

        topwords = topwords[:, :n_topwords_to_use] # Get the top n_topwords words
        self._load_embedding_dict_from_disc() 

        topword_embeddings = []
        for topic in topwords:
            for word in topic:
                topic_embeddings = []
                if word in self.embedding_dict:
                    topic_embeddings.append(self.embedding_dict[word])
                else:
                    embedding = self.word_embedding_model.encode(word)
                    topword_embeddings.append(embedding)
                    self.embedding_dict[word] = embedding
                topic_embeddings.append(embedding)

        if cache_embeddings:
            self._save_embedding_dict_to_disc()
        topic_embeddings = np.array(topic_embeddings)

        return topic_embeddings