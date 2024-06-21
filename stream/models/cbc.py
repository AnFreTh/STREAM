import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.cbc_utils import DocumentCoherence, get_top_tfidf_words_per_document
from ..utils.dataset import TMDataset
from .base import BaseModel


class CBC(BaseModel):
    def __init__(self):
        """
        Initializes the DocumentClusterer with a DataFrame of coherence scores.

        Parameters:
            coherence_scores (DataFrame): DataFrame containing coherence scores between documents.
        """
        self.trained = False

    def get_info(self):
        """
        Get information about the model.

        Returns
        -------
        dict
            Dictionary containing model information including model name
        """
        info = {
            "model_name": "CBC",
            "trained": self.trained,
        }
        return info

    def _create_coherence_graph(self):
        """
        Initializes the CBC model.

        Attributes
        ----------
        trained : bool
            Indicator whether the model has been trained.
        """
        G = nx.Graph()
        for i in self.coherence_scores.index:
            for j in self.coherence_scores.columns:
                # Add an edge if coherence score is above a certain threshold
                # Threshold can be adjusted
                if self.coherence_scores.at[i, j] > 0:
                    G.add_edge(i, j, weight=self.coherence_scores.at[i, j])
        return G

    def cluster_documents(self):
        """
        Clusters documents based on coherence scores.

        Returns
        -------
        dict
            A dictionary mapping cluster labels to lists of document indices.
        """
        G = self._create_coherence_graph()
        partition = community_louvain.best_partition(G, weight="weight")

        clusters = {}
        for node, cluster_id in partition.items():
            clusters.setdefault(cluster_id, []).append(node)

        return clusters

    def combine_documents(self, documents, clusters):
        """
        Combines documents within each cluster.

        Parameters
        ----------
        documents : DataFrame
            Original DataFrame of documents.
        clusters : dict
            Dictionary of document clusters.

        Returns
        -------
        DataFrame
            New DataFrame with combined documents.
        """
        combined_docs = []
        for cluster_id, doc_indices in clusters.items():
            combined_text = " ".join(
                documents.iloc[idx]["text"] for idx in doc_indices
            )  # Assuming 'text' column
            combined_docs.append(
                {"cluster_id": cluster_id, "combined_text": combined_text}
            )

        return pd.DataFrame(combined_docs)

    def _prepare_data(
        self,
        dataset,
        remove_stopwords,
        lowercase,
        remove_punctuation,
        remove_numbers,
        lemmatize,
        stem,
        expand_contractions,
        remove_html_tags,
        remove_special_chars,
        remove_accents,
        custom_stopwords,
        detokenize,
    ):
        """
        Prepares the dataset for clustering.

        Parameters
        ----------
        dataset : TMDataset
            Dataset containing the documents.
        """

        current_steps = {
            "remove_stopwords": remove_stopwords,
            "lowercase": lowercase,
            "remove_punctuation": remove_punctuation,
            "remove_numbers": remove_numbers,
            "lemmatize": lemmatize,
            "stem": stem,
            "expand_contractions": expand_contractions,
            "remove_html_tags": remove_html_tags,
            "remove_special_chars": remove_special_chars,
            "remove_accents": remove_accents,
            "custom_stopwords": custom_stopwords,
            "detokenize": detokenize,
        }

        # Check if the preprocessing steps are already applied
        previous_steps = dataset.info.get("preprocessing_steps", {})

        # Filter out steps that have already been applied
        filtered_steps = {
            key: (
                False
                if key in previous_steps and previous_steps[key] == value
                else value
            )
            for key, value in current_steps.items()
        }

        if custom_stopwords:
            filtered_steps["remove_stopwords"] = True

        # Only preprocess if there are steps that need to be applied
        if filtered_steps:
            dataset.preprocess(**filtered_steps)

        self.dataframe = dataset.dataframe
        self.dataframe["tfidf_top_words"] = get_top_tfidf_words_per_document(
            self.dataframe["text"]
        )

    def fit(
        self,
        dataset: TMDataset = None,
        max_topics: int = 20,
        max_iterations: int = 20,
        remove_stopwords: bool = True,
        lowercase: bool = False,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        expand_contractions: bool = True,
        remove_html_tags: bool = True,
        remove_special_chars: bool = True,
        remove_accents: bool = True,
        custom_stopwords=[],
        detokenize: bool = True,
    ):
        """
        Clusters documents based on coherence scores until the number of clusters is
        within a specified threshold.

        Parameters
        ----------
        dataset : TMDataset, optional
            Dataset containing the documents.
        max_topics : int, optional
            Maximum acceptable number of clusters.
        max_iterations : int, optional
            Maximum number of iterations for clustering.

        Raises
        ------
        AssertionError
            If the dataset is not an instance of TMDataset.
        """
        self.max_topics = max_topics
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        print("--- preparing the dataset ---")
        self._prepare_data(
            dataset,
            remove_stopwords,
            lowercase,
            remove_punctuation,
            remove_numbers,
            lemmatize,
            stem,
            expand_contractions,
            remove_html_tags,
            remove_special_chars,
            remove_accents,
            custom_stopwords,
            detokenize,
        )

        iteration = 0
        current_documents = self.dataframe.copy()
        document_indices = [[i] for i in range(len(current_documents))]

        print("--- start training ---")

        while True:
            print(f"Iteration: {iteration}")
            # Calculate coherence scores for the current set of documents
            coherence_scores = DocumentCoherence(
                current_documents, column="tfidf_top_words"
            ).calculate_document_coherence()

            # Cluster the documents based on the current coherence scores
            self.coherence_scores = coherence_scores
            clusters = self.cluster_documents()

            num_clusters = len(clusters)
            print(f"Iteration {iteration}: {num_clusters} clusters formed.")

            # Prepare for the next iteration
            combined_documents = self.combine_documents(current_documents, clusters)
            current_documents = combined_documents
            iteration += 1

            # Update document indices to reflect their new combined form
            new_document_indices = []
            for cluster_ids in clusters.values():
                new_document_indices.append(
                    [document_indices[idx] for idx in cluster_ids]
                )
            document_indices = new_document_indices

            # Check if the number of clusters is within the threshold
            if 2 <= num_clusters <= self.max_topics:
                break
            elif num_clusters < 2:
                print(
                    "Too few clusters formed. Consider changing parameters or input data."
                )
                break

            # Stop if too many iterations to prevent infinite loop
            if iteration > max_iterations:  # You can adjust this limit
                print("Maximum iterations reached. Stopping clustering process.")
                break

        labels = {}
        for cluster_label, doc_indices_group in enumerate(document_indices):
            for doc_indices in doc_indices_group:
                for index in doc_indices:
                    labels[index] = cluster_label

        self.dataframe["predictions"] = self.dataframe.index.map(labels)
        self.labels = np.array(self.dataframe["predictions"])
        if np.isnan(self.labels).sum() > 0:
            # Store the indices of NaN values
            self.dropped_indices = np.where(np.isnan(self.labels))[0]

            # Replace NaN values with -1 in self.labels
            self.labels[np.isnan(self.labels)] = -1
            self.labels += 1

            # Update the 'predictions' column in the dataframe with -1 where NaN was present
            self.dataframe["predictions"] = self.dataframe["predictions"].fillna(-1)
            self.dataframe["predictions"] += 1
            print("--- replaced NaN values with 0 in topics ---")
            print(
                "--- indices of original NaN values stored in self.dropped_indices ---"
            )

        docs_per_topic = self.dataframe.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )
        print("--- Extracting the Topics ---")
        tfidf, count = c_tf_idf(docs_per_topic["text"].values, m=len(self.dataframe))
        self.topic_dict = extract_tfidf_topics(tfidf, count, docs_per_topic, n=10)

        one_hot_encoder = OneHotEncoder(
            sparse=False
        )  # Use sparse=False to get a dense array
        predictions_one_hot = one_hot_encoder.fit_transform(
            self.dataframe[["predictions"]]
        )

        self.topic_word_distribution = tfidf.T
        self.document_topic_distribution = predictions_one_hot.T
        self.trained = True

    def predict(self, texts):
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
        labels = self.clustering_model.predict(reduced_embeddings)
        return labels

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
        return [
            [word for word, _ in self.topic_dict[key][:n_words]]
            for key in self.topic_dict
        ]

    def get_topic_word_matrix(self):
        """
        Retrieve the topic-word distribution matrix.

        Returns
        -------
        numpy.ndarray
            Topic-word distribution matrix.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        return self.topic_word_distribution

    def get_topic_document_matrix(self):
        """
        Retrieve the topic-document distribution matrix.

        Returns
        -------
        numpy.ndarray
            Topic-document distribution matrix.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        return self.topic_document_matrix
