import os
import pickle
import re

import gensim.downloader as api
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import Dataset, random_split

from ..commons.load_steps import load_model_preprocessing_steps
from ..preprocessor import TextPreprocessor
from .data_downloader import DataDownloader, get_data_home


class TMDataset(Dataset, DataDownloader):
    """
    Topic Modeling Dataset containing methods to fetch and preprocess text data.

    Parameters
    ----------
    name : str, optional
        Name of the dataset.

    Attributes
    ----------
    available_datasets : list of str
        List of available datasets.
    name : str
        Name of the dataset.
    dataframe : pd.DataFrame
        DataFrame containing the dataset.
    embeddings : np.ndarray
        Embeddings for the dataset.
    bow : np.ndarray
        Bag of Words representation of the dataset.
    tfidf : np.ndarray
        TF-IDF representation of the dataset.
    tokens : list of list of str
        Tokenized documents.
    texts : list of str
        Preprocessed text data.
    labels : list of str
        Labels for the dataset.
    language : str
        Language of the dataset.
    preprocessing_steps : dict
        Preprocessing steps to apply to the dataset.

    Notes
    -----
    Available datasets:

    - 20NewsGroup
    - BBC_News
    - Stocktwits_GME
    - Reddit_GME'
    - Reuters'
    - Spotify
    - Spotify_most_popular
    - Poliblogs
    - Spotify_least_popular

    Examples
    --------
    >>> from stream_topic.utils.dataset import TMDataset
    >>> dataset = TMDataset()
    >>> dataset.fetch_dataset("20NewsGroup")
    >>> dataset.preprocess(remove_stopwords=True, lowercase=True)
    >>> dataset.get_bow()
    >>> dataset.get_tfidf()
    >>> dataset.get_word_embeddings()
    >>> dataset.dataframe.head()

    """

    def __init__(self, name=None, language="en"):
        super().__init__()

        self.name = name
        self.dataframe = None
        self.embeddings = None
        self.bow = None
        self.tfidf = None
        self.tokens = None
        self.texts = None
        self.labels = None
        self.language = language
        self.preprocessing_steps = self.default_preprocessing_steps()

    def fetch_dataset(self, name: str, dataset_path=None, source: str = "github"):
        """
        Fetch a dataset by name.

        Parameters
        ----------
        name : str
            Name of the dataset to fetch.
        dataset_path : str, optional
            Path to the dataset directory.
        source : str, optional
            Source of the dataset, by default 'github'. Use 'local' if dataset is available in locally. Then, provide the dataset_path.
        """

        if self.name is not None:
            logger.info(
                f"Dataset name already provided while instantiating the class: {self.name}"
            )
            logger.info(
                f"Overwriting the dataset name with the name provided in fetch_dataset: {name}"
            )
            self.name = name
            logger.info(f"Fetching dataset: {name}")
        else:
            self.name = name
            logger.info(f"Fetching dataset: {name}")

        if source == "github" and dataset_path is None:
            # logger.info(f"Fetching dataset from github")
            self.load_custom_dataset_from_url(name)
            data_home = get_data_home()
            dataset_path = os.path.join(
                data_home, "preprocessed_datasets", name)
            self.info = self.get_info(dataset_path)
        elif source == "local" and dataset_path is not None:
            logger.info(f"Fetching dataset from local path")
            self.load_custom_dataset_from_folder(dataset_path)
            self.info = self.get_info(dataset_path)
        elif dataset_path is None:
            logger.info(f"Fetching dataset from package path")
            dataset_path = self.get_package_dataset_path(name)
            if os.path.exists(dataset_path):
                self.load_custom_dataset_from_folder(dataset_path)
                logger.info(f"Dataset loaded successfully from {dataset_path}")
            else:
                logger.error(f"Dataset path {dataset_path} does not exist.")
                raise ValueError(
                    f"Dataset path {dataset_path} does not exist.")
            # self._load_data_to_dataframe()
            self.info = self.get_info(dataset_path)
        else:
            logger.error(
                f"Dataset path {dataset_path} does not exist. Please provide the correct path or use the exiting dataset."
            )
            raise ValueError(
                f"Dataset path {dataset_path} does not exist. Please provide the correct path or use the exiting dataset."
            )

    def _load_data_to_dataframe(self):
        """
        Load data into a pandas DataFrame.
        """
        self.dataframe = pd.DataFrame(
            {
                "tokens": self.get_corpus(),
                "labels": self.get_labels(),
            }
        )
        self.dataframe["text"] = [" ".join(words)
                                  for words in self.dataframe["tokens"]]
        self.texts = self.dataframe["text"].tolist()
        self.labels = self.dataframe["labels"].tolist()

    def create_load_save_dataset(
        self,
        data,
        dataset_name,
        save_dir,
        doc_column=None,
        label_column=None,
        **kwargs,
    ):
        """
        Create, load, and save a dataset.

        Parameters
        ----------
        data : pd.DataFrame or list
            The data to create the dataset from.
        dataset_name : str
            Name of the dataset.
        save_dir : str
            Directory to save the dataset.
        doc_column : str, optional
            Column name for documents if data is a DataFrame.
        label_column : str, optional
            Column name for labels if data is a DataFrame.
        **kwargs : dict
            Additional columns and their values to include in the dataset.

        Returns
        -------
        Preprocessing
            The preprocessed dataset.
        """
        if isinstance(data, pd.DataFrame):
            if doc_column is None:
                raise ValueError(
                    "doc_column must be specified for DataFrame input")
            documents = [
                self.clean_text(str(row[doc_column])) for _, row in data.iterrows()
            ]
            labels = (
                data[label_column].tolist() if label_column else [
                    None] * len(documents)
            )
        elif isinstance(data, list):
            documents = [self.clean_text(doc) for doc in data]
            labels = [None] * len(documents)
        else:
            raise TypeError(
                "data must be a pandas DataFrame or a list of documents")

        # Initialize preprocessor with kwargs
        preprocessor = TextPreprocessor(**kwargs)
        preprocessed_documents = preprocessor.preprocess_documents(documents)
        self.texts = preprocessed_documents
        self.labels = labels

        # Add additional columns from kwargs to the DataFrame
        additional_columns = {
            key: value for key, value in kwargs.items() if key != "preprocessor"
        }
        additional_columns.update({"text": self.texts, "labels": self.labels})
        self.dataframe = pd.DataFrame(additional_columns)

        # Save the dataset to Parquet format
        if not os.path.exists(save_dir):
            logger.info(f"Dataset save directory does not exist: {save_dir}")
            logger.info(f"Creating directory: {save_dir}")
            os.makedirs(save_dir)

        local_parquet_path = os.path.join(save_dir, f"{dataset_name}.parquet")
        self.dataframe.to_parquet(local_parquet_path)
        logger.info(f"Dataset saved to {local_parquet_path}")

        # Save dataset information
        dataset_info = {
            "name": dataset_name,
            "language": self.language,
            "preprocessing_steps": {
                k: v
                for k, v in preprocessor.__dict__.items()
                if k not in ["stop_words", "language", "contractions_dict"]
            },
        }
        info_path = os.path.join(save_dir, f"{dataset_name}_info.pkl")
        with open(info_path, "wb") as info_file:
            pickle.dump(dataset_info, info_file)
        logger.info(f"Dataset info saved to {info_path}")
        # return preprocessor

    def preprocess(self, model_type=None, custom_stopwords=None, **preprocessing_steps):
        """
        Preprocess the dataset.

        Parameters
        ----------
        model_type : str, optional
            The model type to load the preprocessing steps for.
        custom_stopwords : list of str, optional
            Custom stopwords to remove.
        **preprocessing_steps : dict
            Preprocessing steps to apply

        Returns
        -------
        None
            This method modifies the object's texts and dataframe attributes in place.

        Notes
        -----
        This function applies a series of preprocessing steps to the text data stored in
        the object's `texts` attribute. The preprocessed text is then stored back into the
        `texts` attribute and updated in the `dataframe["text"]` column.
        """
        if model_type:
            preprocessing_steps = load_model_preprocessing_steps(model_type)
        previous_steps = self.preprocessing_steps

        # Filter out steps that have already been applied
        filtered_steps = {
            key: (
                False
                if key in previous_steps and previous_steps[key] == value
                else value
            )
            for key, value in preprocessing_steps.items()
        }

        if custom_stopwords:
            filtered_steps["remove_stopwords"] = True
            filtered_steps["custom_stopwords"] = list(set(custom_stopwords))
        else:
            filtered_steps["custom_stopwords"] = []

        # Only preprocess if there are steps that need to be applied

        if filtered_steps:
            try:
                preprocessor = TextPreprocessor(
                    language=self.language,
                    **preprocessing_steps,
                )
                self.texts = preprocessor.preprocess_documents(self.texts)
                self.dataframe["text"] = self.texts
                self.dataframe["tokens"] = self.dataframe["text"].apply(
                    lambda x: x.split()
                )

                self.info.update(
                    {
                        "preprocessing_steps": {
                            k: v
                            for k, v in preprocessor.__dict__.items()
                            if k != "stopwords"
                        }
                    }
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error in dataset preprocessing: {e}") from e
        self.update_preprocessing_steps(**filtered_steps)

    def update_preprocessing_steps(self, **preprocessing_steps):
        """
        Update preprocessing steps to True if they were previously False.

        Parameters
        ----------
        preprocessing_steps : dict
            Key-value pairs of preprocessing steps to update.
        """
        for step, value in preprocessing_steps.items():
            if (
                value is True
                and step in self.preprocessing_steps
                and not self.preprocessing_steps[step]
            ):
                self.preprocessing_steps[step] = True
            elif value is True and step not in self.preprocessing_steps:
                self.preprocessing_steps[step] = True

    def get_info(self, dataset_path=None):
        """
        Load and return the dataset information.

        Parameters
        ----------
        name : str
            Name of the dataset.
        save_dir : str
            Directory where the dataset is saved.

        Returns
        -------
        dict
            Dictionary containing the dataset information.
        """
        if dataset_path is None:
            dataset_path = self.get_package_dataset_path(self.name)
        elif os.path.exists(dataset_path):
            pass
        else:
            raise ValueError(f"Dataset path {dataset_path} does not exist.")

        info_path = os.path.join(dataset_path, f"{self.name}_info.pkl")
        if os.path.exists(info_path):
            with open(info_path, "rb") as info_file:
                dataset_info = pickle.load(info_file)
            return dataset_info
        else:
            raise FileNotFoundError(
                f"Dataset info file {info_path} does not exist.")

    @staticmethod
    def clean_text(text):
        """
        Clean the input text.

        Parameters
        ----------
        text : str
            Input text to clean.

        Returns
        -------
        str
            Cleaned text.
        """
        text = text.replace("\n", " ").replace("\r", " ").replace("\\", "")
        text = re.sub(r"[{}[\]-]", "", text)
        text = text.encode("utf-8", "replace").decode("utf-8")
        return text

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        dict
            Sample at the given index.
        """
        item = {"text": self.texts[idx]}
        if self.labels[idx] is not None:
            item["label"] = self.labels[idx]
        if self.embeddings is not None:
            item["embedding"] = self.embeddings[idx]
        if self.bow is not None:
            item["bow"] = self.bow[idx]
        if self.tokens is not None:
            item["tokens"] = self.tokens[idx]
        if self.tfidf is not None:
            item["tfidf"] = self.tfidf[idx]
        return item

    def get_corpus(self):
        """
        Get the corpus (tokens) from the dataframe.

        Returns
        -------
        list of list of str
            Corpus tokens.
        """
        return self.dataframe["tokens"].tolist()

    def get_vocabulary(self):
        """
        Get the vocabulary from the dataframe.

        Returns
        -------
        list of str
            Vocabulary.
        """
        # Flatten the list of lists and convert to set for unique words
        all_tokens = [
            token for sublist in self.dataframe["tokens"].tolist() for token in sublist
        ]
        return list(set(all_tokens))

    def get_labels(self):
        """
        Get the labels from the dataframe.

        Returns
        -------
        list of str
            Labels.
        """
        return self.dataframe["labels"].tolist()

    def split_dataset(self, train_ratio=0.8, val_ratio=0.2, seed=None):
        """
        Split the dataset into train, validation, and test sets.

        Parameters
        ----------
        train_ratio : float, optional
            Ratio of the training set, by default 0.8.
        val_ratio : float, optional
            Ratio of the validation set, by default 0.1.
        test_ratio : float, optional
            Ratio of the test set, by default 0.1.
        seed : int, optional
            Random seed for shuffling, by default None.

        Returns
        -------
        tuple of Dataset
            Train, validation, and test datasets.
        """
        total_size = len(self)

        if train_ratio < 0 or val_ratio < 0:
            raise ValueError("Train, val and test ratios must be positive")

        if train_ratio + val_ratio != 1.0:
            raise ValueError("Train, validation and test ratios must sum to 1")

        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size

        if seed is not None:
            np.random.seed(seed)

        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset

    def get_bow(self, **kwargs):
        """
        Get the Bag of Words representation of the corpus.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional arguments to pass to CountVectorizer.

        Returns
        -------
        scipy.sparse.csr_matrix
            BOW matrix.
        list of str
            Feature names.
        """
        corpus = [" ".join(tokens) for tokens in self.get_corpus()]
        vectorizer = CountVectorizer(**kwargs)
        self.bow = vectorizer.fit_transform(
            corpus).toarray().astype(np.float32)
        return self.bow, vectorizer.get_feature_names_out()

    def get_tfidf(self, **kwargs):
        """
        Get the TF-IDF representation of the corpus.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional arguments to pass to TfidfVectorizer.

        Returns
        -------
        scipy.sparse.csr_matrix
            TF-IDF matrix.
        list of str
            Feature names.
        """
        corpus = [" ".join(tokens) for tokens in self.get_corpus()]
        vectorizer = TfidfVectorizer(**kwargs)
        self.tfidf = vectorizer.fit_transform(corpus).toarray()
        return self.tfidf, vectorizer.get_feature_names_out()

    def has_word_embeddings(self, model_name):
        """
        Check if word embeddings are available for the dataset.

        Parameters
        ----------
        model_name : str
            Name of the pre-trained model.

        Returns
        -------
        bool
            True if word embeddings are available, False otherwise.
        """
        return self.has_embeddings(model_name, "word_embeddings")

    def get_word_embeddings(self, model_name="glove-wiki-gigaword-100", vocab=None):
        """
        Get the word embeddings for the vocabulary using a pre-trained model.

        Parameters
        ----------
        model_name : str, optional
            Name of the pre-trained model to use, by default 'glove-wiki-gigaword-100'.
        vocab : list of str, optional

        Returns
        -------
        dict
            Dictionary mapping words to their embeddings.
        """

        assert model_name in [
            "glove-wiki-gigaword-100",
            "paraphrase-MiniLM-L3-v2",
        ], f"model name {model_name} not supported. Can be 'glove-wiki-gigaword-100' and 'paraphrase-MiniLM-L3-v2'"

        if vocab is None:
            vocabulary = self.get_vocabulary()
        else:
            vocabulary = vocab

        if self.has_word_embeddings(model_name):
            return self.get_embeddings(model_name, "word_embeddings")

        if model_name == "glove_wiki_gigaword_100":
            # Load pre-trained model
            model = api.load(model_name)

            embeddings = {word: model[word]
                          for word in vocabulary if word in model}

        if model_name == "paraphrase-MiniLM-L3-v2":
            model = SentenceTransformer(model_name)
            vocabulary = list(vocabulary)
            embeddings = model.encode(
                vocabulary, convert_to_tensor=True, show_progress_bar=True
            )

            embeddings = {word: embeddings[i]
                          for i, word in enumerate(vocabulary)}

            assert len(embeddings) == len(
                vocabulary
            ), "Embeddings and vocabulary length mismatch"

        return embeddings
