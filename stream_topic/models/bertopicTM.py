from datetime import datetime
import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from ..commons.check_steps import check_dataset_steps
from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.mixins import SentenceEncodingMixin
import pandas as pd

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "BERTopicTM"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class BERTopicTM(BaseModel, SentenceEncodingMixin):
    """
    A topic modeling class that uses K-Means clustering on text data.

    This class inherits from the AbstractModel class and utilizes sentence embeddings,
    UMAP for dimensionality reduction, and K-Means for clustering text data into topics.

    Attributes:
        hyperparameters (dict): A dictionary of hyperparameters for the model.
        n_topics (int): The number of topics to cluster the documents into.
        embedding_model (SentenceTransformer): The sentence embedding model.
        umap_args (dict): Arguments for UMAP dimensionality reduction.
        kmeans_args (dict): Arguments for the KMeans clustering algorithm.
        optim (bool): Flag to enable optimization of the number of clusters.
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        umap_args: dict = None,
        min_cluster_size: int = None,
        hdbscan_args: dict = None,
        random_state: int = None,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        **kwargs,
    ):
        """
        Initializes the KmeansTM model with specified parameters.

        Parameters:
            hyperparameters (dict): Model hyperparameters. Defaults to an empty dict.
            num_topics (int): Number of topics. Defaults to 20.
            embedding_model (SentenceTransformer): Sentence embedding model. Defaults to "all-MiniLM-L6-v2".
            umap_args (dict): UMAP arguments. Defaults to an empty dict.
            kmeans_args (dict): KMeans arguments. Defaults to an empty dict.
            random_state (int): Random state for reproducibility. Defaults to None.
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
        if random_state is not None:
            self.umap_args["random_state"] = random_state
        self.min_cluster_size = min_cluster_size
        self.hdbscan_args = self.hparams.get("hdscan_args", hdbscan_args or {})

        self.hparams["umap_args"] = self.umap_args
        self.hparams["hdbscan_args"] = self.hdbscan_args

        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path

        self.save_embeddings = save_embeddings
        self.n_topics = None

        self._status = TrainingStatus.NOT_STARTED
        self.stopwords_path = kwargs.get("stopwords_path", None)

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
            "embedding_model": self.embedding_model_name,
            "umap_args": self.umap_args,
            "hdbscan_args": self.hdbscan_args,
            "trained": self._status.name,
        }
        return info

    def _clustering(self, min_cluster_size=None):
        """
        Applies HDBSCAN clustering to the reduced embeddings.

        Parameters
        ----------
        min_cluster_size : int, optional
            Overrides the configured HDBSCAN ``min_cluster_size`` for this call.
            Used to re-cluster with a smaller value when the initial run finds
            fewer than the target number of topics.
        """

        import importlib

        hdbscan = importlib.import_module("hdbscan")

        assert (
            hasattr(self, "reduced_embeddings") and self.reduced_embeddings is not None
        ), "Reduced embeddings must be generated before clustering."

        try:
            logger.info("--- Creating document cluster ---")
            hdbscan_args = dict(self.hdbscan_args)
            if min_cluster_size is not None:
                hdbscan_args["min_cluster_size"] = int(min_cluster_size)
            self.clustering_model = hdbscan.HDBSCAN(**hdbscan_args)
            self.clustering_model.fit(self.reduced_embeddings)
            # Copy before shifting: mutating clustering_model.labels_ in place
            # would corrupt the -1 noise label that _compute_wcss relies on.
            self.labels = self.clustering_model.labels_.copy()
            if self.labels.min() < 0:
                self.labels += 1

        except Exception as e:
            raise RuntimeError(f"Error in clustering: {e}") from e

        # topic_centroids is only consumed by stream_topic/visuals; it is computed
        # lazily via the topic_centroids property below instead of eagerly on every
        # _clustering call (which the re-cluster retry loop invokes repeatedly).

    @property
    def topic_centroids(self):
        """Mean full-dimensional embedding per cluster (lazy; used by visuals only)."""
        if getattr(self, "_topic_centroids", None) is None:
            labels = np.array(self.labels)
            self._topic_centroids = [
                np.mean(self.embeddings[labels == label], axis=0)
                for label in np.unique(labels)
            ]
        return self._topic_centroids

    def _reduce_topics(self, n_topics):
        """
        Hierarchically merge topics until n_topics remain.

        At each step, the two most similar topics (by c-TF-IDF cosine
        similarity) are merged: their documents are reassigned to the
        smaller label, and c-TF-IDF is recomputed.

        Parameters
        ----------
        n_topics : int
            Target number of topics.
        """
        current_k = len(np.unique(self.labels))

        # If HDBSCAN found fewer than the target number of clusters, we cannot
        # merge down to K. Re-cluster with progressively smaller min_cluster_size
        # until at least K clusters are found (BERTopic's sanctioned route to a
        # target K), then fall through to the merge-down step. This preserves the
        # equal-K guarantee across all models. HDBSCAN's default min_cluster_size
        # is 5, so we start just below whatever was used and shrink toward 2.
        if current_k < n_topics:
            start = int(self.hdbscan_args.get("min_cluster_size", 5))
            for mcs in range(max(start - 1, 2), 1, -1):
                logger.info(
                    f"--- HDBSCAN found {current_k} < {n_topics} clusters; "
                    f"re-clustering with min_cluster_size={mcs} ---"
                )
                self._clustering(min_cluster_size=mcs)
                current_k = len(np.unique(self.labels))
                if current_k >= n_topics:
                    break
            if current_k < n_topics:
                logger.warning(
                    f"--- HDBSCAN could not reach {n_topics} clusters "
                    f"(max found: {current_k}); reporting {current_k} topics ---"
                )

        if current_k <= n_topics:
            return

        logger.info(
            f"--- Reducing {current_k} topics to {n_topics} via hierarchical merging ---"
        )

        # Incremental c-TF-IDF merge (bit-identical to recomputing c_tf_idf every
        # iteration, but tokenizes the corpus ONCE instead of once per merge).
        #
        # Why identical: CountVectorizer's vocabulary is the union of all corpus
        # tokens, which is invariant to how documents are grouped into clusters,
        # and its feature order is sorted, so the (k x V) count matrix has the same
        # columns each iteration. Merging two clusters is exactly summing their two
        # integer count rows -- the per-word column totals (sum_t, hence idf) are
        # therefore invariant across merges, so idf is computed once. tf and tf-idf
        # are then reproduced from the maintained counts with the same numpy ops as
        # c_tf_idf, giving bit-identical similarities and argmax tie-breaking.
        from sklearn.feature_extraction.text import CountVectorizer

        self.dataframe["predictions"] = self.labels
        docs_per_topic = self.dataframe.groupby(
            ["predictions"], as_index=False
        ).agg({"text": " ".join})
        m = len(self.dataframe)

        count = CountVectorizer(ngram_range=(1, 1), stop_words="english").fit(
            docs_per_topic["text"].values
        )
        counts = count.transform(docs_per_topic["text"].values).toarray()
        # groupby(sort=True) => rows aligned to ascending prediction labels.
        labels_arr = docs_per_topic["predictions"].values.copy()

        # idf is invariant across merges (see above): compute once.
        sum_t = counts.sum(axis=0)
        sum_t = np.maximum(sum_t, 1)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        idf[~np.isfinite(idf)] = 0

        while len(labels_arr) > n_topics:
            # Reproduce c_tf_idf's tf / tf-idf from the maintained count matrix.
            t = counts
            w = t.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                tf = np.divide(t.T, w)
                tf[~np.isfinite(tf)] = 0
            tf_idf = np.multiply(tf, idf)
            tf_idf = np.nan_to_num(tf_idf, nan=0.0, posinf=0.0, neginf=0.0)

            # Cosine similarity between topic c-TF-IDF vectors
            sim = cosine_similarity(tf_idf.T)
            np.fill_diagonal(sim, -1)

            # Find the most similar pair
            i, j = np.unravel_index(sim.argmax(), sim.shape)
            merge_from_idx = max(i, j)
            merge_into_idx = min(i, j)
            merge_from = labels_arr[merge_from_idx]
            merge_into = labels_arr[merge_into_idx]

            # Merge cluster rows (exact integer sum) and drop the merged label,
            # keeping labels_arr sorted so row/col order matches a fresh groupby.
            counts[merge_into_idx] += counts[merge_from_idx]
            counts = np.delete(counts, merge_from_idx, axis=0)
            labels_arr = np.delete(labels_arr, merge_from_idx)

            # Merge: reassign all docs from merge_from -> merge_into
            self.labels[self.labels == merge_from] = merge_into

        # Relabel to contiguous 0..n_topics-1. unique_labels == labels_arr (both
        # sorted), so relabeled topic i corresponds exactly to counts row i.
        unique_labels = np.unique(self.labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels = np.array([label_map[l] for l in self.labels])

        # Stash the final c-TF-IDF state so fit() can reuse it on the english path
        # instead of re-tokenizing the whole corpus a second time. counts/idf/vocab
        # here are exactly what a fresh c_tf_idf over the merged clusters produces
        # (vocab is grouping-invariant; counts are exact integer row-sums), so the
        # reconstructed tf_idf and the CountVectorizer feature order are bit-identical.
        w = counts.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = np.divide(counts.T, w)
            tf[~np.isfinite(tf)] = 0
        merged_tfidf = np.multiply(tf, idf)
        merged_tfidf = np.nan_to_num(merged_tfidf, nan=0.0, posinf=0.0, neginf=0.0)
        self._merged_tfidf = merged_tfidf     # (vocab, n_topics)
        self._merged_count = count            # fitted CountVectorizer (english)

        logger.info(f"--- Topic reduction complete: {len(unique_labels)} topics ---")

    def fit(self, dataset, n_topics=None, language="en"):
        """
        Trains the BERTOPIC topic model on the provided dataset.

        Applies sentence embedding, UMAP dimensionality reduction, and hdbscan clustering
        to the dataset to identify distinct topics within the text data. When n_topics is
        provided, topics are hierarchically merged to the target count.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.
        n_topics : int, optional
            Target number of topics. If provided, HDBSCAN topics are merged
            down to this count. If None, uses HDBSCAN's natural clustering.
        """

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."
        if language == 'chinese':
            check_dataset_steps(dataset, logger, MODEL_NAME, language='chinese')
        else:
            check_dataset_steps(dataset, logger, MODEL_NAME)
        self._status = TrainingStatus.INITIALIZED

        # Resync from hparams: HPO writes tuned UMAP/HDBSCAN args into hparams, but
        # dim-reduction/clustering read the instance attributes set in __init__.
        # Without this the 5-seed HPO refit silently uses default UMAP/HDBSCAN.
        self.umap_args = self.hparams.get("umap_args", self.umap_args)
        self.hdbscan_args = self.hparams.get("hdbscan_args", self.hdbscan_args)

        # Reset any stashed merge-loop c-TF-IDF from a previous fit/HPO refit so a
        # stale reuse can never leak across fits.
        self._merged_tfidf = None
        self._merged_count = None

        if self.stopwords_path is not None:
            with open(self.stopwords_path, 'r', encoding='UTF-8') as f:
                stop_words = [line.strip() for line in f]
                stopwords = pd.DataFrame({'w': stop_words})
            stopwords_list = set(stopwords['w'])
            try:
                logger.info(f"--- Training {MODEL_NAME} topic model ---")
                self._status = TrainingStatus.RUNNING
                self.dataset, self.embeddings = self.prepare_embeddings(dataset, logger)
                self.dataframe = self.dataset.dataframe
                self.reduced_embeddings = self.dim_reduction(logger)

                self._clustering()

                # Reduce to target n_topics if specified
                if n_topics is not None:
                    self._reduce_topics(n_topics)

                self.dataframe["predictions"] = self.labels
                docs_per_topic = self.dataframe.groupby(
                    ["predictions"], as_index=False
                ).agg({"text": " ".join})

                tfidf, count = c_tf_idf(
                    docs_per_topic["text"].values, m=len(self.dataframe),stop_words=stopwords_list
                )

                self.topic_dict = extract_tfidf_topics(tfidf, count, docs_per_topic, n=100)

                one_hot_encoder = OneHotEncoder(sparse_output=False)
                predictions_one_hot = one_hot_encoder.fit_transform(
                    self.dataframe[["predictions"]]
                )

                self.beta = tfidf
                self.theta = predictions_one_hot
            except Exception as e:
                logger.error(f"Error in training: {e}")
                self._status = TrainingStatus.FAILED
                raise
            except KeyboardInterrupt:
                logger.error("Training interrupted.")
                self._status = TrainingStatus.INTERRUPTED
                raise
        else:
            try:
                logger.info(f"--- Training {MODEL_NAME} topic model ---")
                self._status = TrainingStatus.RUNNING
                self.dataset, self.embeddings = self.prepare_embeddings(dataset, logger)
                self.dataframe = self.dataset.dataframe
                self.reduced_embeddings = self.dim_reduction(logger)

                self._clustering()

                # Reduce to target n_topics if specified
                if n_topics is not None:
                    self._reduce_topics(n_topics)

                self.dataframe["predictions"] = self.labels

                # If _reduce_topics ran, it left the final c-TF-IDF (over merged
                # clusters, english stopwords, full corpus) in self._merged_tfidf /
                # self._merged_count -- bit-identical to what the block below would
                # recompute via a second groupby-join + full-corpus tokenization.
                # Reuse it and skip the redundant work. Rows in _merged_tfidf.T
                # correspond to the CONTIGUOUS relabeled topics 0..K-1 (labels_arr
                # was kept in sorted order == np.unique(self.labels)).
                if self._merged_tfidf is not None and self._merged_count is not None:
                    tfidf, count = self._merged_tfidf, self._merged_count
                    labels_sorted = np.unique(self.dataframe["predictions"].values)
                    docs_per_topic = pd.DataFrame({"predictions": labels_sorted})
                else:
                    docs_per_topic = self.dataframe.groupby(
                        ["predictions"], as_index=False
                    ).agg({"text": " ".join})
                    tfidf, count = c_tf_idf(
                        docs_per_topic["text"].values, m=len(self.dataframe)
                    )

                self.topic_dict = extract_tfidf_topics(tfidf, count, docs_per_topic, n=100)

                # theta is one-hot over contiguous cluster labels 0..K-1 (see
                # _reduce_topics relabel). np.eye(K)[labels] is bit-identical to
                # OneHotEncoder's dense output when categories are contiguous.
                labels_int = self.dataframe["predictions"].to_numpy()
                if labels_int.min() >= 0 and labels_int.max() == len(np.unique(labels_int)) - 1:
                    predictions_one_hot = np.eye(int(labels_int.max()) + 1, dtype=float)[labels_int]
                else:
                    one_hot_encoder = OneHotEncoder(sparse_output=False)
                    predictions_one_hot = one_hot_encoder.fit_transform(
                        self.dataframe[["predictions"]]
                    )

                self.beta = tfidf
                self.theta = predictions_one_hot
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
        self.n_topics = len(self.topic_dict)

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
        labels = self.clustering_model.approximate_predict(reduced_embeddings)
        return labels

    def suggest_hyperparameters(self, trial):
        """
        Suggests hyperparameters for the model using an Optuna trial.

        This method uses an Optuna trial object to suggest a set of hyperparameters for the model.
        The suggested hyperparameters are stored in the `hparams` dictionary of the model.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The Optuna trial object used for suggesting hyperparameters.
        """
        # Suggest UMAP parameters
        self.hparams["umap_args"]["n_neighbors"] = trial.suggest_int(
            "n_neighbors", 10, 50
        )
        self.hparams["umap_args"]["n_components"] = trial.suggest_int(
            "n_components", 5, 50
        )
        self.hparams["umap_args"]["metric"] = trial.suggest_categorical(
            "metric", ["cosine", "euclidean"]
        )

        # Suggest HDBSCAN parameters
        self.hparams["hdbscan_args"]["min_cluster_size"] = trial.suggest_int(
            "min_cluster_size", 5, 100
        )
        self.hparams["hdbscan_args"]["min_samples"] = trial.suggest_int(
            "min_samples", 1, 100
        )
        self.hparams["hdbscan_args"]["cluster_selection_epsilon"] = trial.suggest_float(
            "cluster_selection_epsilon", 0.0, 1.0
        )

        self.umap_args = self.hparams.get("umap_args")
        self.hdbscan_args = self.hparams.get("hdbscan_args")

    def optimize_and_fit(
        self,
        dataset,
        min_topics=2,
        max_topics=20,
        criterion="aic",
        n_trials=100,
        custom_metric=None,
        timeout=None,
    ):
        """
        A new method in the child class that calls the parent class's optimize_hyperparameters method.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.
        min_topics : int, optional
            Minimum number of topics to evaluate, by default 2.
        max_topics : int, optional
            Maximum number of topics to evaluate, by default 20.
        criterion : str, optional
            Criterion to use for optimization ('aic', 'bic', or 'custom'), by default 'aic'.
        n_trials : int, optional
            Number of trials for optimization, by default 100.
        custom_metric : object, optional
            Custom metric object with a `score` method for evaluation, by default None.

        Returns
        -------
        dict
            Dictionary containing the best parameters and the optimal number of topics.
        """
        best_params = super().optimize_hyperparameters(
            dataset=dataset,
            min_topics=min_topics,
            max_topics=max_topics,
            criterion=criterion,
            n_trials=n_trials,
            custom_metric=custom_metric,
            timeout=timeout,
        )

        return best_params

    def _compute_wcss(self):
        """
        Compute the within-cluster sum of squares (WCSS) for HDBSCAN clusters.

        Returns
        -------
        float
            The WCSS value.
        """
        wcss = 0.0
        labels = self.clustering_model.labels_
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            cluster_points = self.reduced_embeddings[labels == label]
            centroid = cluster_points.mean(axis=0)
            wcss += ((cluster_points - centroid) ** 2).sum()
        return wcss

    def calculate_aic(self, n_topics=None):
        """
        Calculate the AIC for the HDBSCAN model.

        Returns
        -------
        float
            AIC score.
        """
        wcss = self._compute_wcss()
        n = self.reduced_embeddings.shape[0]
        k = len(np.unique(self.clustering_model.labels_))
        return n * np.log(wcss / n) + 2 * k

    def calculate_bic(self, n_topics=None):
        """
        Calculate the BIC for the HDBSCAN model.

        Returns
        -------
        float
            BIC score.
        """
        wcss = self._compute_wcss()
        n = self.reduced_embeddings.shape[0]
        k = len(np.unique(self.clustering_model.labels_))
        return n * np.log(wcss / n) + k * np.log(n)
