import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from datetime import datetime
import pandas as pd

from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus

MODEL_NAME = "NMFTM"
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)

class NMFTM(BaseModel):
    """
    A topic modeling class that uses Non-negative Matrix Factorization (NMF) to cluster text data into topics.

    This class inherits from the BaseModel class and utilizes TF-IDF for vectorization and NMF for dimensionality reduction and clustering.

    Attributes
    ----------
    n_topics : int
        Number of topics to extract.
    max_features : int
        Maximum number of features used for vectorization.
    nmf_args : dict
        Arguments for NMF clustering.
    tfidf_args : dict
        Arguments for TF-IDF vectorization.
    tfidf_vectorizer : TfidfVectorizer
        TF-IDF vectorizer instance.
    nmf_model : NMF
        NMF model instance.
    trained : bool
        Flag indicating whether the model has been trained.

    Methods
    -------
    get_info()
        Returns a dictionary containing information about the model.
    fit(dataset)
        Trains the model on the provided dataset and extracts topics.
    get_topics(n_words=10)
        Retrieves the top words for each topic.
    predict(texts)
        Predicts topics for new documents.
    """
    def __init__(self, n_topics=20, max_features=5000, nmf_args=None, tfidf_args=None):
        super().__init__()
        self.n_topics = n_topics
        self.max_features = max_features
        self.tfidf_args = tfidf_args or {"max_df": 0.95, "min_df": 2, "max_features": max_features}
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_args)
        self.nmf_args = nmf_args or {}
        self.nmf_model = NMF(n_components=n_topics, init='random', random_state=42, **self.nmf_args)
        self._status = TrainingStatus.NOT_STARTED

    def get_info(self):
        """
        Get information about the model.

        Returns
        -------
        dict
            Dictionary containing model information including model name, number of topics, vectorization and clustering arguments, and training status.
        """
        info = {
            "model_name": MODEL_NAME,
            "n_topics": self.n_topics,
            "nmf_args": self.nmf_args,
            "tfidf_args": self.tfidf_args,
            "trained_status": self._status.name
        }
        return info

    def fit(self, dataset: TMDataset):
        """
        Trains the NMF topic model on the provided dataset.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.

        Raises
        ------
        RuntimeError
            If the training fails due to an error.
        """
        assert isinstance(dataset, TMDataset), "The dataset must be an instance of TMDataset."
        self._status = TrainingStatus.RUNNING
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(dataset.texts)
            W = self.nmf_model.fit_transform(tfidf_matrix) # Document-topic matrix (Theta)
            H = self.nmf_model.components_  # Topic-term matrix (Beta)

            # Assigning attributes 
            self.labels = np.argmax(W, axis=1)
            self.theta = W  
            self.beta = H
            
            # Prepare data for visualization
            topic_data = pd.DataFrame(columns=['predictions', 'text'])
            for i in range(self.nmf_model.n_components_):
                topic_texts = [dataset.texts[j] for j, z in enumerate(W[:, i]) if z > 0.1]
                if not topic_texts:
                    continue
                aggregated_texts = ' '.join(topic_texts)
                new_row = pd.DataFrame({'predictions': [i], 'text': [aggregated_texts]})
                topic_data = pd.concat([topic_data, new_row], ignore_index=True)

            if topic_data.empty:
                raise RuntimeError("No topics were extracted, model training failed.")

            tfidf, count = c_tf_idf(topic_data['text'].tolist(), len(dataset.texts))
            self.topic_dict = extract_tfidf_topics(tfidf, count, topic_data)
            self._status = TrainingStatus.SUCCEEDED
        except Exception as e:
            logger.error(f"Error in training: {e}")
            self._status = TrainingStatus.FAILED
            raise RuntimeError(f"Training failed with an error: {e}")

    def get_topics(self, n_words=10):
        """
        Retrieves the top words for each topic from the extracted topics dictionary.

        Parameters
        ----------
        n_words : int
            Number of top words to retrieve per topic.

        Returns
        -------
        dict
            A dictionary of topics with their corresponding list of top words.
        """
        if not hasattr(self, 'topic_dict'):
            raise RuntimeError("Model has not been trained yet or training failed.")
        return {topic: words[:n_words] for topic, words in self.topic_dict.items()}

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
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        W = self.nmf_model.transform(tfidf_matrix)
        return np.argmax(W, axis=1)


