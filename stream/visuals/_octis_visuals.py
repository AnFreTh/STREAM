import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class OctisWrapperVisualModel:
    """
    A wrapper class for visualizing OCTIS models with additional functionalities such as embedding extraction and topic dictionary creation.

    Attributes:
        octis_model (AbstractModel): The OCTIS model to be wrapped.
        octis_output (dict): The output from the OCTIS model, expected to contain "topic-document-matrix" and "topic-word-matrix".
        dataset (TMDataset): The dataset used for the OCTIS model, for embedding extraction and other purposes.
        embedding_model_name (str): Name of the model to be used for sentence embeddings via Sentence Transformers.
        embeddings_folder_path (str, optional): Path to the folder where precomputed embeddings are stored. If not provided, embeddings will be computed on the fly.
        embeddings_file_path (str, optional): Path to the specific file where precomputed embeddings are stored.

    Methods:
        get_topic_dict(top_words=20): Creates a dictionary mapping each topic to its top words and their corresponding weights.
        get_embeddings(): Extracts or loads embeddings for the dataset based on the provided embedding model.
    """

    def __init__(
        self,
        octis_model,
        octis_output,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
    ):
        """
        Initializes the OctisWrapperVisualModel with the given parameters.

        Args:
            octis_model (AbstractModel): The OCTIS model to be wrapped.
            octis_output (dict): The output from the OCTIS model.
            dataset (TMDataset): The dataset used for the OCTIS model.
            embedding_model_name (str): Name of the model for sentence embeddings. Defaults to "all-MiniLM-L6-v2".
            embeddings_folder_path (str, optional): Path to the folder with precomputed embeddings. Defaults to None.
            embeddings_file_path (str, optional): Path to the file with precomputed embeddings. Defaults to None.
        """

        super().__init__()
        self._model = octis_model
        self.output = octis_output
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path
        self.labels = np.argmax(self.output["topic-document-matrix"], axis=0)
        self.trained = True

    def get_topic_dict(self, top_words=20):
        """
        Generates a dictionary of topics with their top words and corresponding weights.

        Args:
            top_words (int): The number of top words to retrieve for each topic. Defaults to 20.

        Populates:
            self.output["topic_dict"] (dict): A dictionary where keys are topic indices and values are lists of tuples (word, weight).
        """
        result = {}
        if top_words > 0:
            topics_output = []
            for idx, topic in enumerate(self.output["topic-word-matrix"]):
                top_k = np.argsort(topic)[-top_words:]
                vals = np.sort(topic)[-top_words:]
                top_k_words = list(
                    reversed([self._model.id2word[i] for i in top_k]))
                topics_output.append(top_k_words)
                result[idx] = [(word, val)
                               for word, val in zip(top_k_words, vals)]
        self.topic_dict = result
        self.trained = True

    def get_embeddings(self, dataset):
        """
        Extracts or loads embeddings for the dataset texts. The embeddings are either loaded from a specified path or computed on the fly.

        Populates:
            self.embeddings (np.ndarray): The embeddings for the dataset texts.
            self.dataframe (pd.DataFrame): A dataframe containing the dataset texts and possibly other information.
        """
        if not self._model.use_partitions:
            self.embeddings = dataset.get_embeddings(
                self.embedding_model_name,
                self.embeddings_path,
                self.embeddings_file_path,
            )
            self.dataframe = self.dataset.dataframe

        else:
            self.dataframe = pd.DataFrame(
                {
                    "tokens": dataset.get_partitioned_corpus()[0],
                }
            )
            self.dataframe["text"] = [
                " ".join(words) for words in self.dataframe["tokens"]
            ]

            self.embeddings = self.embedding_model.encode(
                self.dataframe["text"], show_progress_bar=False
            )

    def get_topics(self):
        return self.output["topics"]
