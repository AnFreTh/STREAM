import umap.umap_ as umap
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
from ..utils.topic_extraction import TopicExtractor
from ..utils.cleaning import clean_topics
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from ..data_utils.dataset import TMDataset

data_dir = "../preprocessed_datasets"


class CEDC(AbstractModel):
    """
    A topic modeling class that utilizes sentence embeddings, UMAP for dimensionality
    reduction, and Gaussian Mixture Models (GMM) for clustering text data into topics.

    This class inherits from the AbstractModel class and is designed for clustering
    and topic extraction from textual data.

    Attributes:
        hyperparameters (dict): A dictionary of hyperparameters for the model.
        n_topics (int): The number of topics to identify in the dataset.
        embedding_model (SentenceTransformer): The sentence embedding model used to
            convert text to embeddings.
        umap_args (dict): Arguments for UMAP dimensionality reduction.
        gmm_args (dict): Arguments for the Gaussian Mixture Model.
        dataset (pandas.DataFrame): The dataset used for training, containing the text documents.
    """

    def __init__(
        self,
        num_topics: int = 20,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        umap_args: dict = {},
        random_state: int = None,
        gmm_args: dict = {},
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
    ):
        """
        Initializes the CEDC model with specified hyperparameters, number of topics,
        embedding model, UMAP arguments, and additional options.

        Parameters:
            num_topics (int, optional): The number of topics to identify in the dataset.
                Defaults to 20.
            embedding_model_name (str, optional): The name of the SentenceTransformer model
                used for generating sentence embeddings. Defaults to "all-MiniLM-L6-v2".
            umap_args (dict, optional): A dictionary containing arguments for UMAP dimensionality
                reduction. Defaults to an empty dictionary.
            random_state (int, optional): Random seed for UMAP (if provided). Defaults to None.
            gmm_args (dict, optional): A dictionary containing arguments for the Gaussian Mixture Model.
                Defaults to an empty dictionary.
            embeddings_folder_path (str, optional): Path to the folder containing precomputed embeddings.
                Defaults to None.
            embeddings_file_path (str, optional): Path to the precomputed embeddings file. Defaults to None.
        """
        super().__init__()
        self.trained = False
        self.n_topics = num_topics
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path
        self.umap_args = (
            umap_args
            if umap_args
            else {
                "n_neighbors": 15,
                "n_components": 15,
                "metric": "cosine",
            }
        )
        if random_state is not None:
            self.umap_args["random_state"] = random_state

        if gmm_args:
            self.gmm_args = gmm_args
        else:
            self.gmm_args = {
                "n_components": self.n_topics,
                "covariance_type": "full",
                "tol": 0.001,
                "reg_covar": 0.000001,
                "max_iter": 100,
                "n_init": 1,
                "init_params": "kmeans",
            }

    def _prepare_data(self):
        """
        Prepares the dataset for clustering.

        """

        assert hasattr(self, "dataset") and isinstance(
            self.dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        self.embeddings = self.dataset.get_embeddings(
            self.embedding_model_name, self.embeddings_path, self.embeddings_file_path
        )
        self.dataframe = self.dataset.dataframe

    def _clustering(self):
        """
        Applies GMM clustering to the reduced embeddings.
        """
        assert (
            hasattr(self, "reduced_embeddings") and self.reduced_embeddings is not None
        ), "Reduced embeddings must be generated before clustering."

        try:
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
        """
        assert (
            hasattr(self, "embeddings") and self.embeddings is not None
        ), "Embeddings must be generated before dimensionality reduction."

        try:
            self.reducer = umap.UMAP(**self.umap_args)
            self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
        except Exception as e:
            raise ValueError(f"Error in dimensionality reduction: {e}")

    def _get_topic_document_matrix(self):
        assert (
            self.trained
        ), "Model must be trained before accessing the topic-document matrix."
        # Safely get the topic-document matrix with a default value of None if not found
        return self.output.get("topic-document-matrix", None)

    def train_model(
        self,
        dataset,
        only_nouns=False,
        clean=False,
        clean_threshold=0.85,
        expansion_corpus="octis",
        n_words=20,
    ):
        """
        Trains the CEDC model on the provided dataset, performing topic extraction
        and clustering using Gaussian Mixture Models.

        Parameters:
            dataset: The dataset to train the model on, containing text documents.
            only_nouns (bool, optional): If true, only extracts nouns for topic
                modeling. Defaults to False.
            clean (bool, optional): If true, performs cleaning of the extracted topics
                based on a similarity threshold. Defaults to False.
            clean_threshold (float, optional): The similarity threshold used for
                cleaning topics. Defaults to 0.85.
            expansion_corpus (str, optional): The name of the corpus used for topic
                expansion. Defaults to "octis".
            n_words (int, optional): The number of words to consider in each topic.
                Defaults to 20.

        Returns:
            dict: A dictionary containing the extracted topics, and potentially cleaned
            topics and centroids.
        """

        self.dataset = dataset
        print("--- preparing the dataset ---")
        self._prepare_data()
        print("--- Dimensionality Reduction ---")
        self._dim_reduction()
        print("--- Training the model ---")
        self._clustering()

        assert (
            hasattr(self, "soft_labels") and self.soft_labels is not None
        ), "Clustering must generate labels."

        TE = TopicExtractor(
            dataset=self.dataset,
            topic_assignments=self.soft_labels,
            n_topics=self.n_topics,
            embedding_model=self.embedding_model,
        )

        print("--- extract topics ---")
        topics, self.topic_centroids = TE._noun_extractor_haystack(
            self.embeddings,
            n=n_words + 20,
            corpus=expansion_corpus,
            only_nouns=only_nouns,
        )

        if clean:
            cleaned_topics, cleaned_centroids = clean_topics(
                topics, similarity=clean_threshold, embedding_model=self.embedding_model
            )
            topics = cleaned_topics
            self.topic_centroids = cleaned_centroids

        words_list = []
        new_topics = {}
        for k in range(self.n_topics):
            words = [
                word
                for t in topics[k][0:n_words]
                for word in t
                if isinstance(word, str)
            ]
            weights = [
                weight
                for t in topics[k][0:10]
                for weight in t
                if isinstance(weight, float)
            ]
            weights = [weight / sum(weights) for weight in weights]
            new_topics[k] = list(zip(words, weights))
            words_list.append(words)

        self.output = {}
        self.output["topics"] = words_list
        self.output["topic-word-matrix"] = None
        self.output["topic_dict"] = topics
        self.output["topic-document-matrix"] = np.array(self.soft_labels.T)

        self.trained = True

        return self.output
