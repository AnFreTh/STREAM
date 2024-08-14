from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger
from sklearn.mixture import GaussianMixture
import os
from ..commons.check_steps import check_dataset_steps
from ..preprocessor._embedder import BaseEmbedder, GensimBackend
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "WordCluTM"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)
WORD_EMBEDDING_MODEL_NAME = (
    "paraphrase-MiniLM-L3-v2"  # use this model for word embeddings for now
)


class WordCluTM(BaseModel):
    """
    A topic modeling class that uses Word2Vec embeddings and K-Means or GMM clustering on vocabulary to form coherent word clusters.
    """

    def __init__(
        self,
        umap_args: dict = None,
        random_state: int = None,
        gmm_args: dict = None,
        train_word_embeddings: bool = True,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        word_embedding_model_name: str = WORD_EMBEDDING_MODEL_NAME,
        save_embeddings: bool = False,
        **kwargs,
    ):
        """
        Initializes the WordCluTM model with specified parameters.

        Args:
            umap_args (dict, optional): Parameters for UMAP dimensionality reduction. Defaults to a pre-defined dictionary if not provided.
            random_state (int, optional): Seed for random number generation to ensure reproducibility. Defaults to None.
            gmm_args (dict, optional): Parameters for Gaussian Mixture Model (GMM) clustering. Defaults to a pre-defined dictionary if not provided.
            train_word_embeddings (bool, optional): Flag indicating whether to train Word2Vec embeddings or use pre-trained embeddings. Defaults to True.
            embeddings_folder_path (str, optional): Path to the folder where word embeddings should be saved. Defaults to None.
            embeddings_file_path (str, optional): Path to the file containing pre-trained word embeddings. Defaults to None.
            word_embedding_model_name (str, optional): The name of the pre-trained model to be used for word embeddings. Defaults to 'paraphrase-MiniLM-L3-v2'.
            save_embeddings (bool, optional): Flag indicating whether to save the trained word embeddings. Defaults to False.
            **kwargs: Additional keyword arguments passed to the BaseModel initialization.
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
        self.n_topics = None

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

        self.hparams["umap_args"] = self.umap_args
        self.hparams["gmm_args"] = self.gmm_args

        self.word_embeddings_path = embeddings_folder_path
        self.word_embedding_model_name = word_embedding_model_name
        self.word_embeddings_file_path = embeddings_file_path
        self.save_word_embeddings = save_embeddings

        self._status = TrainingStatus.NOT_STARTED

        self.word_embeddings_prepared = False
        self.train_word_embeddings = train_word_embeddings
        self.optimize = False

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
            "umap_args": self.umap_args,
            "trained": self._status.name,
        }
        return info

    def train_word2vec(
        self, sentences, epochs, vector_size, window, min_count, workers, logger
    ):
        """
        Train a Word2Vec model on the given sentences.

        Args:
            sentences (list): List of tokenized sentences.
        """
        # Initialize Word2Vec model
        self.word2vec_model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
        )

        # Build the vocabulary from the sentences
        self.word2vec_model.build_vocab(sentences)

        logger.info(f"--- Train Word2Vec ---")
        # Train the Word2Vec model
        self.word2vec_model.train(
            sentences, total_examples=len(sentences), epochs=epochs
        )

        # Initialize BaseEmbedder with GensimBackend
        self.base_embedder = BaseEmbedder(GensimBackend(self.word2vec_model.wv))

    def _prepare_word_embeddings(self, dataset, logger):
        """
        Prepare the word embeddings for the dataset.

        Parameters
        ----------
        data_module : TMDataModule
            The data module used for training. This contains the actually used vocabulary after preprocessing.
        dataset : TMDataset
            The dataset to be used for training.
        logger : Logger
            The logger to log messages.
        """

        if dataset.has_word_embeddings(self.word_embedding_model_name):
            logger.info(
                f"--- Loading precomputed {self.word_embedding_model_name} word embeddings ---"
            )
            self.word_embeddings = dataset.get_word_embeddings(
                self.word_embedding_model_name,
                self.word_embeddings_path,
                self.word_embeddings_file_path,
            )

        else:
            logger.info(
                f"--- Creating {self.word_embedding_model_name} word embeddings ---"
            )
            self.word_embeddings = dataset.get_word_embeddings(
                model_name=self.word_embedding_model_name,
                vocab=dataset.get_vocabulary(),  # use the vocabulary from the data module
            )
            if (
                self.save_word_embeddings
                and self.word_embeddings_path is not None
                and not os.path.exists(self.word_embeddings_path)
            ):
                os.makedirs(self.word_embeddings_path)
            if self.save_word_embeddings:
                dataset.save_word_embeddings(
                    word_embeddings=self.word_embeddings,
                    model_name=self.word_embedding_model_name,
                    path=self.word_embeddings_path,
                    file_name=self.word_embeddings_file_path,
                )

        self.word_embeddings_prepared = True

    def _clustering(self):
        """
        Applies GMM clustering to the reduced Word embeddings.

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
            logger.info("--- Creating Word cluster ---")
            self.GMM = GaussianMixture(
                **self.gmm_args,
            ).fit(self.reduced_embeddings)

            gmm_predictions = self.GMM.predict_proba(self.reduced_embeddings)
            self.beta = pd.DataFrame(gmm_predictions)

        except Exception as e:
            raise RuntimeError(f"Error in clustering: {e}") from e

    def fit(
        self,
        dataset: TMDataset = None,
        n_topics: int = 20,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        n_words=10,
        word2vec_epochs=100,
    ):

        self.vector_size = vector_size

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        check_dataset_steps(dataset, logger, MODEL_NAME)

        self.n_topics = n_topics
        if self.n_topics <= 0:
            raise ValueError("Number of topics must be greater than 0.")

        sentences = dataset.get_corpus()
        self._status = TrainingStatus.INITIALIZED

        unique_words = list(set(word for sentence in sentences for word in sentence))

        try:
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            if self.train_word_embeddings:
                self.train_word2vec(
                    sentences=sentences,
                    epochs=word2vec_epochs,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=workers,
                    logger=logger,
                )  # Train Word2Vec model

                self.embeddings = np.array(
                    [
                        (
                            self.word2vec_model.wv[word]
                            if word in dataset.get_vocabulary()
                            else np.zeros(vector_size)
                        )
                        for word in unique_words
                    ]
                )

            else:
                self._prepare_word_embeddings(dataset, logger)
                self.embeddings = np.stack(list(self.word_embeddings.values()))
                if self.embeddings[0].shape != self.vector_size:
                    self.vector_size = self.embeddings[0].shape

            self.reduced_embeddings = self.dim_reduction(logger)
            self._clustering()

            doc_topic_distributions = []
            self.doc_embeddings = []
            logger.info(f"--- Compute doc embeddings ---")
            for doc in sentences:
                # Collect word embeddings for the document
                if self.train_word_embeddings:
                    word_embeddings = [
                        self.word2vec_model.wv[word]
                        for word in doc
                        if word in self.word2vec_model.wv
                    ]
                else:
                    word_embeddings = [
                        np.array(self.word_embeddings[word]) for word in doc
                    ]
                # Compute the mean embedding for the document if there are valid word embeddings
                if word_embeddings:
                    self.doc_embeddings.append(np.mean(word_embeddings, axis=0))
                else:
                    # Append a zero array if no valid word embeddings are found
                    self.doc_embeddings.append(np.zeros(self.vector_size))

            # Replace any NaN values in the final list with zero arrays
            self.doc_embeddings = [
                arr if not np.isnan(arr).any() else np.zeros(arr.shape)
                for arr in self.doc_embeddings
            ]
            if len(self.doc_embeddings) > 0:
                # Reduce the dimensionality of the document embedding
                reduced_doc_embedding = self.reducer.transform(self.doc_embeddings)
                # Predict the topic distribution for the reduced document embedding
                doc_topic_distribution = self.GMM.predict_proba(reduced_doc_embedding)
                # Add the topic distribution to the list
                doc_topic_distributions.append(doc_topic_distribution[0])

            self.theta = pd.DataFrame(doc_topic_distributions)

            # Create topic_dict
            self.topic_dict = {}
            for topic_idx in range(self.beta.shape[1]):
                # Get the indices of the words sorted by their probability of belonging to this topic, in descending order
                sorted_indices = np.argsort(self.beta.iloc[:, topic_idx])[::-1]
                # Get the top n_words for this topic based on the sorted indices
                top_words = [
                    (unique_words[i], self.beta.iloc[i, topic_idx])
                    for i in sorted_indices[:n_words]
                ]
                # Store the top words and their probabilities in the dictionary
                self.topic_dict[topic_idx] = top_words

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

    def predict(self, texts):
        pass

    def suggest_hyperparameters(self, trial):
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

        # Suggest GMM parameters
        self.hparams["gmm_args"]["covariance_type"] = trial.suggest_categorical(
            "covariance_type", ["full", "tied", "diag", "spherical"]
        )
        self.hparams["gmm_args"]["tol"] = trial.suggest_float(
            "tol", 1e-4, 1e-1, log=True
        )
        self.hparams["gmm_args"]["reg_covar"] = trial.suggest_float(
            "reg_covar", 1e-6, 1e-3, log=True
        )
        self.hparams["gmm_args"]["max_iter"] = trial.suggest_int("max_iter", 100, 1000)
        self.hparams["gmm_args"]["n_init"] = trial.suggest_int("n_init", 1, 10)
        self.hparams["gmm_args"]["init_params"] = trial.suggest_categorical(
            "init_params", ["kmeans", "random"]
        )

        self.umap_args = self.hparams.get("umap_args")
        self.gmmargs = self.hparams.get("gmm_args")

    def optimize_and_fit(
        self,
        dataset,
        min_topics=2,
        max_topics=20,
        criterion="aic",
        n_trials=100,
        custom_metric=None,
    ):
        """
        A new method in the child class that optimizes and fits the model.

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
        )

        return best_params

    def calculate_aic(self, n_topics=None):

        return self.GMM.aic(self.reduced_embeddings)

    def calculate_bic(self, n_topics=None):

        return self.GMM.bic(self.reduced_embeddings)
