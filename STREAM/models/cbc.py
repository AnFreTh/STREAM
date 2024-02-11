from octis.models.model import AbstractModel
from ..data_utils.dataset import TMDataset
import networkx as nx
import community as community_louvain
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from ..utils.tf_idf import c_tf_idf, extract_tfidf_topics
from ..utils.cbc_utils import DocumentCoherence, get_top_tfidf_words_per_document


class CBC(AbstractModel):
    def __init__(self, max_topics: int = 20):
        """
        Initializes the DocumentClusterer with a DataFrame of coherence scores.

        Parameters:
            coherence_scores (DataFrame): DataFrame containing coherence scores between documents.
        """
        self.trained = False
        self.max_topics = max_topics

    def _create_coherence_graph(self):
        """
        Creates a graph from the coherence scores where nodes represent documents
        and edges represent coherence scores.
        """
        G = nx.Graph()
        for i in self.coherence_scores.index:
            for j in self.coherence_scores.columns:
                # Add an edge if coherence score is above a certain threshold
                if self.coherence_scores.at[i, j] > 0:  # Threshold can be adjusted
                    G.add_edge(i, j, weight=self.coherence_scores.at[i, j])
        return G

    def cluster_documents(self):
        """
        Clusters documents based on coherence scores.

        Returns:
            dict: A dictionary mapping cluster labels to lists of document indices.
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

        Parameters:
            documents (DataFrame): Original DataFrame of documents.
            clusters (dict): Dictionary of document clusters.

        Returns:
            DataFrame: New DataFrame with combined documents.
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

    def _prepare_data(self):
        """
        Prepares the dataset for clustering.

        """

        self.dataset.get_dataframe()
        self.dataframe = self.dataset.dataframe
        self.dataframe["tfidf_top_words"] = get_top_tfidf_words_per_document(
            self.dataframe["text"]
        )

    def _get_topic_document_matrix(self):
        assert (
            self.trained
        ), "Model must be trained before accessing the topic-document matrix."
        # Safely get the topic-document matrix with a default value of None if not found
        return self.output.get("topic-document-matrix", None)

    def train_model(self, dataset):
        """
        Clusters documents based on coherence scores until the number of clusters is
        within a specified threshold.

        Parameters:
            documents (DataFrame): DataFrame containing the documents.
            threshold (int): Maximum acceptable number of clusters.

        Returns:
            DataFrame: DataFrame containing the final combined documents in each cluster.
        """
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."
        self.dataset = dataset
        print("--- preparing the dataset ---")
        self._prepare_data()

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
            if iteration > 20:  # You can adjust this limit
                print("Maximum iterations reached. Stopping clustering process.")
                break

        self.trained = True
        self.labels = {}
        for cluster_label, doc_indices_group in enumerate(document_indices):
            for doc_indices in doc_indices_group:
                for index in doc_indices:
                    self.labels[index] = cluster_label

        self.dataframe["predictions"] = self.dataframe.index.map(self.labels)
        docs_per_topic = self.dataframe.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )
        print("--- Extracting the Topics ---")
        tfidf, count = c_tf_idf(docs_per_topic["text"].values, m=len(self.dataframe))
        topics = extract_tfidf_topics(tfidf, count, docs_per_topic, n=10)

        one_hot_encoder = OneHotEncoder(
            sparse=False
        )  # Use sparse=False to get a dense array
        predictions_one_hot = one_hot_encoder.fit_transform(
            self.dataframe[["predictions"]]
        )

        # Transpose the one-hot encoded matrix to get shape (k, n)
        topic_document_matrix = predictions_one_hot.T

        self.output = {
            "topics": [[word for word, _ in topics[key]] for key in topics],
            "topic-word-matrix": tfidf.T,
            "topic_dict": topics,
            "topic-document-matrix": topic_document_matrix,  # Include the transposed one-hot encoded matrix
        }
        self.trained = True
        return self.output
