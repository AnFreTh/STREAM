import json
import os
import pickle
import re

import gensim.downloader as api
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import DataLoader, Dataset, random_split

from ..preprocessor import TextPreprocessor


class TMDataset(Dataset):
    def __init__(self, name=None, language="en"):
        """
        Initialize the TMDataset.

        Parameters
        ----------
        name : str, optional
            Name of the dataset.
        """
        super().__init__()
        self.dataset_registry = [
            "20NewsGroup",
            "M10",
            "Spotify",
            "Spotify_most_popular",
            "Poliblogs",
            "Reuters",
            "BBC_News",
            "DBLP",
            "DBPedia_IT",
            "Europarl_IT",
        ]
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

    def default_preprocessing_steps(self):
        return {
            "remove_stopwords": False,
            "lowercase": True,
            "remove_punctuation": False,
            "remove_numbers": False,
            "lemmatize": False,
            "stem": False,
            "expand_contractions": True,
            "remove_html_tags": True,
            "remove_special_chars": True,
            "remove_accents": False,
            "custom_stopwords": set(),
            "detokenize": True,
        }

    def load_model_preprocessing_steps(self, model_type, filepath=None):
        """
        Load the default preprocessing steps from a JSON file.

        Parameters
        ----------
        filepath : str
            The path to the JSON file containing the default preprocessing steps.

        Returns
        -------
        dict
            The default preprocessing steps.
        """
        if filepath is None:
            # Determine the absolute path based on the current file's location
            current_dir = os.path.dirname(__file__)
            filepath = os.path.join(
                current_dir, "..", "preprocessor", "default_preprocessing_steps.json"
            )
            filepath = os.path.abspath(filepath)

        with open(filepath, "r") as file:
            all_steps = json.load(file)
        return all_steps.get(model_type, {})

    def fetch_dataset(self, name, dataset_path=None):
        """
        Fetch a dataset by name.

        Parameters
        ----------
        name : str
            Name of the dataset to fetch.
        dataset_path : str, optional
            Path to the dataset directory.
        """
        self.name = name
        if dataset_path is None:
            dataset_path = self.get_package_dataset_path(name)
        if os.path.exists(dataset_path):
            self.load_custom_dataset_from_folder(dataset_path)
        else:
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        # self._load_data_to_dataframe()

        self.info = self.get_info(dataset_path)

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
        self.dataframe["text"] = [" ".join(words) for words in self.dataframe["tokens"]]
        self.texts = self.dataframe["text"].tolist()
        self.labels = self.dataframe["labels"].tolist()

    def get_package_dataset_path(self, name):
        """
        Get the path to the package dataset.

        Parameters
        ----------
        name : str
            Name of the dataset.

        Returns
        -------
        str
            Path to the dataset.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        my_package_dir = os.path.dirname(script_dir)
        dataset_path = os.path.join(my_package_dir, "preprocessed_datasets", name)
        return dataset_path

    def has_embeddings(self, embedding_model_name, path=None, file_name=None):
        """
        Check if embeddings are available for the dataset.

        Parameters
        ----------
        embedding_model_name : str
            Name of the embedding model used.
        path : str, optional
            Path where embeddings are expected to be saved.
        file_name : str, optional
            File name for the embeddings.

        Returns
        -------
        bool
            True if embeddings are available, False otherwise.
        """
        if path is None:
            path = self.get_package_embeddings_path(self.name)
        embeddings_file = (
            os.path.join(path, file_name)
            if file_name
            else os.path.join(
                path, f"{self.name}_embeddings_{embedding_model_name}.pkl"
            )
        )
        return os.path.exists(embeddings_file)

    def save_embeddings(
        self, embeddings, embedding_model_name, path=None, file_name=None
    ):
        """
        Save embeddings for the dataset.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to save.
        embedding_model_name : str
            Name of the embedding model used.
        path : str, optional
            Path to save the embeddings.
        file_name : str, optional
            File name for the embeddings.
        """
        if path is None:
            path = self.get_package_embeddings_path(self.name)
        embeddings_file = (
            os.path.join(path, file_name)
            if file_name
            else os.path.join(
                path, f"{self.name}_embeddings_{embedding_model_name}.pkl"
            )
        )
        with open(embeddings_file, "wb") as file:
            pickle.dump(embeddings, file)

    def get_embeddings(self, embedding_model_name, path=None, file_name=None):
        """
        Get embeddings for the dataset.

        Parameters
        ----------
        embedding_model_name : str
            Name of the embedding model to use.
        path : str, optional
            Path to save the embeddings.
        file_name : str, optional
            File name for the embeddings.

        Returns
        -------
        np.ndarray
            Embeddings for the dataset.
        """
        if not self.has_embeddings(embedding_model_name, path, file_name):
            raise ValueError(
                "Embeddings are not available. Run the encoding process first or load embeddings."
            )

        # print("--- Loading pre-computed document embeddings ---")

        if self.embeddings is None:
            if path is None:
                path = self.get_package_embeddings_path(self.name)
            embeddings_file = (
                os.path.join(path, file_name)
                if file_name
                else os.path.join(
                    path, f"{self.name}_embeddings_{embedding_model_name}.pkl"
                )
            )
            with open(embeddings_file, "rb") as file:
                self.embeddings = pickle.load(file)

        return self.embeddings

    def get_package_embeddings_path(self, name):
        """
        Get the path to the package embeddings.

        Parameters
        ----------
        name : str
            Name of the dataset.

        Returns
        -------
        str
            Path to the embeddings.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        my_package_dir = os.path.dirname(script_dir)
        dataset_path = os.path.join(my_package_dir, "pre_embedded_datasets", name)
        return dataset_path

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
                raise ValueError("doc_column must be specified for DataFrame input")
            documents = [
                self.clean_text(str(row[doc_column])) for _, row in data.iterrows()
            ]
            labels = (
                data[label_column].tolist() if label_column else [None] * len(documents)
            )
        elif isinstance(data, list):
            documents = [self.clean_text(doc) for doc in data]
            labels = [None] * len(documents)
        else:
            raise TypeError("data must be a pandas DataFrame or a list of documents")

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
        parquet_path = os.path.join(save_dir, f"{dataset_name}.parquet")
        self.dataframe.to_parquet(parquet_path)

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

        return preprocessor

    def preprocess(self, model_type=None, custom_stopwords=None, **preprocessing_steps):
        """
        Preprocess the dataset.

        Parameters
        ----------
        language : str, optional
            The language to use for preprocessing (default is "english").
        remove_stopwords : bool, optional
            Whether to remove stopwords (default is False).
        lowercase : bool, optional
            Whether to convert text to lowercase (default is True).
        remove_punctuation : bool, optional
            Whether to remove punctuation (default is True).
        remove_numbers : bool, optional
            Whether to remove numbers (default is True).
        lemmatize : bool, optional
            Whether to lemmatize words (default is False).
        stem : bool, optional
            Whether to stem words (default is False).
        expand_contractions : bool, optional
            Whether to expand contractions (default is False).
        remove_html_tags : bool, optional
            Whether to remove HTML tags (default is False).
        remove_special_chars : bool, optional
            Whether to remove special characters (default is False).
        remove_accents : bool, optional
            Whether to remove accents (default is False).
        custom_stopwords : list of str, optional
            List of custom stopwords to remove (default is an empty list).
        detokenize : bool, optional
            Whether to detokenize the text after processing (default is True).

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
            preprocessing_steps = self.load_model_preprocessing_steps(model_type)
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
                raise RuntimeError(f"Error in dataset preprocessing: {e}") from e
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
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Dataset info file {info_path} does not exist.")

        with open(info_path, "rb") as info_file:
            dataset_info = pickle.load(info_file)

        return dataset_info

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

    def load_custom_dataset_from_folder(self, dataset_path):
        """
        Load a custom dataset from a folder.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset folder.
        """
        parquet_path = os.path.join(dataset_path, f"{self.name}.parquet")
        if os.path.exists(parquet_path):
            self.load_dataset_from_parquet(parquet_path)
        else:
            documents_path = os.path.join(dataset_path, "corpus.txt")
            labels_path = os.path.join(dataset_path, "labels.txt")

            with open(documents_path, encoding="utf-8") as f:
                documents = f.readlines()

            with open(labels_path, encoding="utf-8") as f:
                labels = f.readlines()

            self.dataframe = pd.DataFrame(
                {
                    "text": [doc.strip() for doc in documents],
                    "labels": [label.strip() for label in labels],
                }
            )

            self.dataframe["tokens"] = self.dataframe["text"].apply(lambda x: x.split())
            self.texts = self.dataframe["text"].tolist()
            self.labels = self.dataframe["labels"].tolist()

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

        if train_ratio == 0 and val_ratio == 0:
            raise ValueError("Train, val and test ratios cannot all be 0")

        if train_ratio + val_ratio != 1.0:
            raise ValueError("Train, validation and test ratios must sum to 1")

        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)

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
        self.bow = vectorizer.fit_transform(corpus).toarray().astype(np.float32)
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

    def get_word_embeddings(self, model_name="glove-wiki-gigaword-100"):
        """
        Get the word embeddings for the vocabulary using a pre-trained model.

        Parameters
        ----------
        model_name : str, optional
            Name of the pre-trained model to use, by default 'glove-wiki-gigaword-100'.

        Returns
        -------
        dict
            Dictionary mapping words to their embeddings.
        """

        if model_name == "glove_wiki_gigaword_100":
            # Load pre-trained model
            model = api.load(model_name)

            vocabulary = self.get_vocabulary()
            embeddings = {word: model[word] for word in vocabulary if word in model}

        if model_name == "paraphrase-MiniLM-L3-v2":
            model = SentenceTransformer(encoder_model)
            embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)

        return embeddings

    def _save_to_parquet(self, save_dir, dataset_name):
        """
        Save the dataset to a Parquet file.

        Parameters
        ----------
        save_dir : str
            Directory to save the dataset.
        dataset_name : str
            Name of the dataset.
        """
        save_path = os.path.join(save_dir, f"{dataset_name}.parquet")
        self.dataframe.to_parquet(save_path, index=False)

    def load_dataset_from_parquet(self, load_path):
        """
        Load a dataset from a Parquet file.

        Parameters
        ----------
        load_path : str
            Path to the Parquet file.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File {load_path} does not exist.")
        self.dataframe = pd.read_parquet(load_path)
        self.dataframe["tokens"] = self.dataframe["text"].apply(lambda x: x.split())
        self.texts = self.dataframe["text"].tolist()
        self.labels = self.dataframe["labels"].tolist()
