from datetime import datetime

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import OneHotEncoder

from ..commons.check_steps import check_dataset_steps
from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.cbc_utils import (DocumentCoherence,
                               get_top_tfidf_words_per_document)
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus

MODEL_NAME = "CBC"
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class CBC(BaseModel):
    def __init__(self):
        """
        Initializes the DocumentClusterer with a DataFrame of coherence scores.

        Parameters:
            coherence_scores (DataFrame): DataFrame containing coherence scores between documents.
        """
        self._status = TrainingStatus.NOT_STARTED
        self.n_topics = None

    def get_info(self):
        """
        Get information about the model.

        Returns
        -------
        dict
            Dictionary containing model information including model name
        """
        info = {
            "model_name": MODEL_NAME,
            "num_topics": self.n_topics,
            "trained": self._status.name,
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
        try:
            logger.info("--- Creating document cluster ---")
            G = self._create_coherence_graph()
            partition = community_louvain.best_partition(G, weight="weight")

            clusters = {}
            for node, cluster_id in partition.items():
                clusters.setdefault(cluster_id, []).append(node)

        except Exception as e:
            raise RuntimeError(f"Error in clustering: {e}") from e

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

    def prepare_data(
        self,
        dataset,
    ):
        """
        Prepares the dataset for clustering.

        Parameters
        ----------
        dataset : TMDataset
            Dataset containing the documents.
        """

        self.dataframe = dataset.dataframe
        self.dataframe["tfidf_top_words"] = get_top_tfidf_words_per_document(
            self.dataframe["text"]
        )

    def fit(
        self,
        dataset: TMDataset = None,
        max_topics: int = 20,
        max_iterations: int = 20,
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

        check_dataset_steps(dataset, logger, MODEL_NAME)
        self.dataset = dataset

        self.prepare_data(
            dataset,
        )

        iteration = 0
        current_documents = self.dataframe.copy()
        document_indices = [[i] for i in range(len(current_documents))]

        self._status = TrainingStatus.INITIALIZED

        try:
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
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
                print(
                    f"Iteration {iteration}: {num_clusters} clusters formed.")

                # Prepare for the next iteration
                combined_documents = self.combine_documents(
                    current_documents, clusters)
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

        except Exception as e:
            logger.error(f"Error in training: {e}")
            self._status = TrainingStatus.FAILED
            raise
        except KeyboardInterrupt:
            logger.error("Training interrupted.")
            self._status = TrainingStatus.INTERRUPTED
            raise

        labels = {}
        for cluster_label, doc_indices_group in enumerate(document_indices):
            for doc_indices in doc_indices_group:
                for index in doc_indices:
                    labels[index] = cluster_label

        self.dataframe["predictions"] = self.dataframe.index.map(labels)
        self.labels = np.array(self.dataframe["predictions"])
        self.n_topics = len(np.unique(self.labels))
        if np.isnan(self.labels).sum() > 0:
            # Store the indices of NaN values
            self.dropped_indices = np.where(np.isnan(self.labels))[0]

            # Replace NaN values with -1 in self.labels
            self.labels[np.isnan(self.labels)] = -1
            self.labels += 1

            # Update the 'predictions' column in the dataframe with -1 where NaN was present
            self.dataframe["predictions"] = self.dataframe["predictions"].fillna(
                -1)
            self.dataframe["predictions"] += 1
            print("--- replaced NaN values with 0 in topics ---")
            print(
                "--- indices of original NaN values stored in self.dropped_indices ---"
            )

        docs_per_topic = self.dataframe.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )
        logger.info("--- Extract topics ---")
        tfidf, count = c_tf_idf(
            docs_per_topic["text"].values, m=len(self.dataframe))
        self.topic_dict = extract_tfidf_topics(
            tfidf, count, docs_per_topic, n=10)

        one_hot_encoder = OneHotEncoder(
            sparse=False
        )  # Use sparse=False to get a dense array
        predictions_one_hot = one_hot_encoder.fit_transform(
            self.dataframe[["predictions"]]
        )

        logger.info("--- Training completed successfully. ---")
        self._status = TrainingStatus.SUCCEEDED

        self.beta = tfidf
        self.theta = predictions_one_hot

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
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        embeddings = self.encode_documents(
            texts, encoder_model=self.embedding_model_name, use_average=True
        )
        reduced_embeddings = self.reducer.transform(embeddings)
        labels = self.clustering_model.predict(reduced_embeddings)
        return labels
