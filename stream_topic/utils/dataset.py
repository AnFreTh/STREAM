import os
import pickle
import re
import jieba
import gensim.downloader as api
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

    def __init__(self, name=None, **kwargs):
        super().__init__()

        self.name = name
        self.dataframe = None
        self.embeddings = None
        self.bow = None
        self.tfidf = None
        self.tokens = None
        self.texts = None
        self.labels = None
        self.features = None
        self.language = kwargs.get("language", "en")
        self.preprocessing_steps = self.default_preprocessing_steps()
        self.stopwords_path = kwargs.get("stopwords_path", None)
        # Store min_df and max_df for BOW/TF-IDF vectorization
        self.min_df = 1
        self.max_df = 1.0

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

        if dataset_path is None:
            # Try package path first
            dataset_path = self.get_package_dataset_path(name)
            if os.path.exists(dataset_path):
                logger.info(f"Fetching dataset from package path")
                self.load_custom_dataset_from_folder(dataset_path)
                logger.info(f"Dataset loaded successfully from {dataset_path}")
                self.info = self.get_info(dataset_path)
            elif source == "github":
                # Fallback to GitHub if not in package
                self.load_custom_dataset_from_url(name)
                data_home = get_data_home()
                dataset_path = os.path.join(data_home, "preprocessed_datasets", name)
                self.info = self.get_info(dataset_path)
            else:
                logger.error(f"Dataset path {dataset_path} does not exist.")
                raise ValueError(f"Dataset path {dataset_path} does not exist.")
        else:
            logger.info(f"Fetching dataset from local path")
            self.load_custom_dataset_from_folder(dataset_path)
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
            key: value for key, value in kwargs.items() if key not in ["preprocessor", "remove_pos"]
        }
        additional_columns.update({"text": self.texts, "labels": self.labels})
        # delete empty ['text'] line
        df = pd.DataFrame(additional_columns)
        new_df = df[df['text'] != '']
        new_df = new_df.reset_index(drop=True) 
        self.dataframe = new_df

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
                if k not in ["stop_words", "language", "contractions_dict","cc",'thu']
                
            },
            "opencc_config": 't2s.json',
        }
        info_path = os.path.join(save_dir, f"{dataset_name}_info.pkl")
        with open(info_path, "wb") as info_file:
            pickle.dump(dataset_info, info_file)
        logger.info(f"Dataset info saved to {info_path}")
        # return preprocessor

    def preprocess(self, model_type=None, custom_stopwords=None, min_word_length=None, min_word_freq=None, 
                   tool='jieba', custom_dict=None, remove_pos=None, domain=None,**preprocessing_steps):
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
            preprocessing_steps = load_model_preprocessing_steps(model_type, language=self.language)
        previous_steps = self.preprocessing_steps

        # Extract and store min_df and max_df for BOW/TF-IDF vectorization
        if "min_df" in preprocessing_steps:
            self.min_df = preprocessing_steps.pop("min_df")
            logger.info(f"Set min_df={self.min_df} for BOW/TF-IDF vectorization")
        if "max_df" in preprocessing_steps:
            self.max_df = preprocessing_steps.pop("max_df")
            logger.info(f"Set max_df={self.max_df} for BOW/TF-IDF vectorization")

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

        # Segmentation / Chinese params flow through the same filtered dict
        if min_word_length is not None:
            filtered_steps['min_word_length'] = min_word_length
        if min_word_freq is not None:
            filtered_steps['min_word_freq'] = min_word_freq
        if tool is not None:
            filtered_steps['segmentation_tool'] = tool
        if custom_dict is not None:
            filtered_steps['segmentation_dict'] = custom_dict
        if remove_pos is not None:
            filtered_steps['remove_pos'] = remove_pos
        if domain is not None:
            filtered_steps['domain'] = domain

        if filtered_steps:
            try:
                preprocessor = TextPreprocessor(
                    language=self.language,
                    stopwords_path=self.stopwords_path,
                    **filtered_steps,
                )
                self.texts = preprocessor.preprocess_documents(self.texts)
                self.dataframe["text"] = self.texts
                # if self.language == "chinese": 
                #     self.dataframe["tokens"] = self.dataframe["text"].apply(lambda x: list(jieba.cut(x)))
                # else:
                self.dataframe["tokens"] = self.dataframe["text"].apply(lambda x: x.split())

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
        if os.path.exists(info_path):
            with open(info_path, "rb") as info_file:
                dataset_info = pickle.load(info_file)
            return dataset_info
        else:
            logger.warning(f"Dataset info file {info_path} not found, using defaults.")
            return {
                "name": self.name,
                "language": self.language,
                "preprocessing_steps": {},
            }

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
        if self.features is not None:
            item["features"] = self.features[idx]
        if hasattr(self, "boc") and self.boc is not None:
            item["boc"] = self.boc[idx]
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

        # random_split draws from torch's global generator, NOT numpy, so seeding
        # numpy here had no effect on the split. Pass an explicit torch Generator
        # so the train/val split is genuinely reproducible from `seed`.
        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator().manual_seed(int(seed))

        train_dataset, val_dataset = random_split(
            self, [train_size, val_size], generator=generator
        )
        return train_dataset, val_dataset

    def get_bow(self, min_df=None, max_df=None, **kwargs):
        """
        Get the Bag of Words representation of the corpus.

        Parameters
        ----------
        min_df : int or float, optional
            Minimum document frequency for CountVectorizer.
            If None, uses the value set during preprocessing (default 1).
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold.
        max_df : float or int, optional
            Maximum document frequency for CountVectorizer.
            If None, uses the value set during preprocessing (default 1.0).
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold.
        **kwargs : dict, optional
            Additional arguments to pass to CountVectorizer.

        Returns
        -------
        scipy.sparse.csr_matrix
            BOW matrix.
        list of str
            Feature names.
        """
        # Use stored values if not explicitly provided
        min_df = min_df if min_df is not None else self.min_df
        max_df = max_df if max_df is not None else self.max_df

        # If BOW already exists and parameters match, return cached version
        if (
            self.bow is not None
            and hasattr(self, "_bow_vocab")
            and hasattr(self, "_bow_min_df")
            and hasattr(self, "_bow_max_df")
            and self._bow_min_df == min_df
            and self._bow_max_df == max_df
        ):
            logger.info(f"Returning cached BOW (vocab size: {len(self._bow_vocab)})")
            return self.bow, self._bow_vocab

        logger.info(f"Creating BOW with min_df={min_df}, max_df={max_df}")

        corpus = [" ".join(tokens) for tokens in self.get_corpus()]
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, **kwargs)
        self.bow = vectorizer.fit_transform(corpus).toarray().astype(np.float32)
        self._bow_vocab = vectorizer.get_feature_names_out()
        self._bow_min_df = min_df
        self._bow_max_df = max_df

        logger.info(f"BOW vocabulary size: {len(self._bow_vocab)}")

        return self.bow, self._bow_vocab

    def get_tfidf(self, min_df=None, max_df=None, **kwargs):
        """
        Get the TF-IDF representation of the corpus.

        Parameters
        ----------
        min_df : int or float, optional
            Minimum document frequency for TfidfVectorizer.
            If None, uses the value set during preprocessing (default 1).
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold.
        max_df : float or int, optional
            Maximum document frequency for TfidfVectorizer.
            If None, uses the value set during preprocessing (default 1.0).
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold.
        **kwargs : dict, optional
            Additional arguments to pass to TfidfVectorizer.

        Returns
        -------
        scipy.sparse.csr_matrix
            TF-IDF matrix.
        list of str
            Feature names.
        """
        # Use stored values if not explicitly provided
        min_df = min_df if min_df is not None else self.min_df
        max_df = max_df if max_df is not None else self.max_df

        logger.info(f"Creating TF-IDF with min_df={min_df}, max_df={max_df}")

        corpus = [" ".join(tokens) for tokens in self.get_corpus()]
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, **kwargs)
        self.tfidf = vectorizer.fit_transform(corpus).toarray()

        logger.info(
            f"TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out())}"
        )

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

    def get_word_embeddings(self, model_name="paraphrase-MiniLM-L3-v2", vocab=None):
        """
        Get the word embeddings for the vocabulary using a pre-trained model.

        Parameters
        ----------
        model_name : str, optional
            Name of the pre-trained model to use, by default 'paraphrase-MiniLM-L3-v2'.
        vocab : list of str, optional
            Vocabulary to get embeddings for. If None, uses dataset vocabulary.

        Returns
        -------
        dict
            Dictionary mapping words to their embeddings.
        """
        model_path = model_name
        if os.path.exists(model_name) and os.path.isdir(model_name):
            model_name = os.path.basename(model_name)
        assert model_name in [
            "glove-wiki-gigaword-100",
            "paraphrase-MiniLM-L3-v2",
            "paraphrase-MiniLM-L6-v2",
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "Conan-embedding-v1",
        ], f"model name {model_name} not supported."

        if vocab is None:
            vocabulary = self.get_vocabulary()
        else:
            vocabulary = vocab

        if self.has_word_embeddings(model_name):
            return self.get_embeddings(model_name, "word_embeddings")

        # Use sentence-transformers for all embeddings
        model = SentenceTransformer(model_name)
        vocabulary = list(vocabulary)
        embeddings_array = model.encode(
            vocabulary, convert_to_tensor=False, show_progress_bar=True
        )

        embeddings = {word: embeddings_array[i] for i, word in enumerate(vocabulary)}

        assert len(embeddings) == len(
            vocabulary
        ), "Embeddings and vocabulary length mismatch"
        return embeddings

    def preprocess_features(self):
        """
        Preprocess the features for the dataset.

        Returns
        -------
        None
            This method modifies the object's features attribute in place.
        """
        self.features = self.dataframe
        # Drop the columns "text" and "tokens" if they exist
        self.features = self.features.drop(
            columns=["text", "tokens", "labels"], errors="ignore"
        )

        # Separate numeric and categorical columns
        numeric_columns = self.features.select_dtypes(
            include=["int64", "float64"]
        ).columns
        categorical_columns = self.features.select_dtypes(
            include=["object", "category"]
        ).columns

        # Standardize numeric features
        if not numeric_columns.empty:
            scaler = StandardScaler()
            self.features[numeric_columns] = scaler.fit_transform(
                self.features[numeric_columns]
            )

        # Integer encode categorical features
        if not categorical_columns.empty:
            for col in categorical_columns:
                encoder = LabelEncoder()
                self.features[col] = encoder.fit_transform(
                    self.features[col].astype(str)
                )

        self.features = self.features.to_numpy(dtype=np.float32)

    def get_semantic_ids(
        self,
        n_levels: int = 3,
        codes_per_level: int = 64,
        embedding_model: str = "paraphrase-MiniLM-L3-v2",
        random_state: int = 42,
        ensure_unique: bool = True,
        cooccurrence_weight: float = 0.0,
        cooccurrence_dim: int = 128,
    ):
        """
        Generate hierarchical Semantic IDs for vocabulary via recursive k-means.

        Each word is mapped to a unique Semantic ID (SID) representing its path 
        through a hierarchical clustering of word embeddings. This enables 
        factorized topic modeling with reduced vocabulary size.

        Parameters
        ----------
        n_levels : int, optional
            Number of hierarchy levels, by default 3.
        codes_per_level : int, optional
            Number of clusters at each level, by default 64.
        embedding_model : str, optional
            Sentence transformer model for word embeddings,
            by default "paraphrase-MiniLM-L3-v2".
        random_state : int, optional
            Random seed for k-means clustering, by default 42.
        ensure_unique : bool, optional
            If True, resolve collisions to guarantee unique SIDs, by default True.
        cooccurrence_weight : float, optional
            Weight for co-occurrence embeddings (0-1). 0 = pure semantic,
            1 = pure co-occurrence, 0.5 = equal blend. Default 0.0.
        cooccurrence_dim : int, optional
            Dimensionality for PPMI-SVD co-occurrence embeddings, by default 128.

        Returns
        -------
        tuple
            (word_sids, boc) where:
            - word_sids: np.ndarray of shape (vocab_size, n_levels)
            - boc: np.ndarray of shape (n_docs, n_levels * codes_per_level)

        Examples
        --------
        >>> dataset.get_bow()  # Must be called first
        >>> word_sids, boc = dataset.get_semantic_ids(n_levels=3, codes_per_level=64)
        >>> # Hybrid: 50% semantic + 50% co-occurrence
        >>> word_sids, boc = dataset.get_semantic_ids(cooccurrence_weight=0.5)
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        from collections import Counter

        # Ensure BOW exists
        if self.bow is None or not hasattr(self, "_bow_vocab"):
            raise ValueError("Must call get_bow() before get_semantic_ids()")

        vocab = list(self._bow_vocab)
        vocab_size = len(vocab)

        logger.info(f"Generating Semantic IDs: {n_levels} levels, {codes_per_level} codes/level")
        logger.info(f"Vocabulary size: {vocab_size}")

        # Get semantic word embeddings
        logger.info(f"Loading word embeddings from {embedding_model}...")
        model = SentenceTransformer(embedding_model)
        semantic_embeddings = model.encode(vocab, show_progress_bar=True)
        semantic_embeddings = normalize(semantic_embeddings)

        # Compute co-occurrence embeddings if weight > 0
        if cooccurrence_weight > 0:
            logger.info(f"Computing PPMI-SVD co-occurrence embeddings (weight={cooccurrence_weight})...")
            cooc_embeddings = self._compute_ppmi_svd_embeddings(
                n_components=cooccurrence_dim, random_state=random_state
            )
            cooc_embeddings = normalize(cooc_embeddings)
            
            # Match dimensions via projection if needed
            if semantic_embeddings.shape[1] != cooc_embeddings.shape[1]:
                target_dim = semantic_embeddings.shape[1]
                np.random.seed(random_state)
                proj = np.random.randn(cooc_embeddings.shape[1], target_dim).astype(np.float32)
                proj /= np.sqrt(cooc_embeddings.shape[1])
                cooc_embeddings = normalize(cooc_embeddings @ proj)
            
            # Blend embeddings
            word_embeddings = (
                (1 - cooccurrence_weight) * semantic_embeddings +
                cooccurrence_weight * cooc_embeddings
            )
            word_embeddings = normalize(word_embeddings)
            logger.info(f"Blended embeddings: {word_embeddings.shape}")
        else:
            word_embeddings = semantic_embeddings

        # Hierarchical k-means clustering
        sids = np.zeros((vocab_size, n_levels), dtype=np.int64)
        leaf_cluster = np.zeros(vocab_size, dtype=np.int64)
        embed_dim = word_embeddings.shape[1]
        
        # Store centroids for each level
        centroids_per_level = []

        for level in range(n_levels):
            logger.info(f"Clustering level {level + 1}/{n_levels}...")

            if level == 0:
                n_clusters = min(codes_per_level, vocab_size)
                kmeans = KMeans(
                    n_clusters=n_clusters, random_state=random_state, n_init=10
                )
                labels = kmeans.fit_predict(word_embeddings)
                sids[:, level] = labels
                leaf_cluster = labels
                
                # Store centroids (pad if fewer clusters than codes_per_level)
                centroids = np.zeros((codes_per_level, embed_dim), dtype=np.float32)
                centroids[:n_clusters] = kmeans.cluster_centers_
                centroids_per_level.append(centroids)
            else:
                new_codes = np.zeros(vocab_size, dtype=np.int64)
                unique_leaves = np.unique(leaf_cluster)
                
                # Aggregate centroids for this level
                level_centroids = np.zeros((codes_per_level, embed_dim), dtype=np.float32)
                centroid_counts = np.zeros(codes_per_level, dtype=np.int32)

                for leaf_id in unique_leaves:
                    mask = leaf_cluster == leaf_id
                    if mask.sum() == 0:
                        continue

                    indices = np.where(mask)[0]
                    subset_embeddings = word_embeddings[indices]

                    n_clusters = min(codes_per_level, len(indices))
                    if n_clusters < 2:
                        new_codes[indices] = 0
                        level_centroids[0] += subset_embeddings.mean(axis=0)
                        centroid_counts[0] += 1
                    else:
                        kmeans = KMeans(
                            n_clusters=n_clusters, random_state=random_state, n_init=10
                        )
                        new_codes[indices] = kmeans.fit_predict(subset_embeddings)
                        # Accumulate centroids
                        for c in range(n_clusters):
                            level_centroids[c] += kmeans.cluster_centers_[c]
                            centroid_counts[c] += 1

                sids[:, level] = new_codes
                leaf_cluster = leaf_cluster * codes_per_level + new_codes
                
                # Average centroids
                for c in range(codes_per_level):
                    if centroid_counts[c] > 0:
                        level_centroids[c] /= centroid_counts[c]
                centroids_per_level.append(level_centroids)

        # Check for collisions and resolve if needed
        if ensure_unique:
            sid_tuples = [tuple(s) for s in sids]
            sid_counts = Counter(sid_tuples)
            collisions = {k: v for k, v in sid_counts.items() if v > 1}
            
            if collisions:
                n_colliding = sum(collisions.values())
                logger.warning(f"Found {len(collisions)} SID collisions affecting {n_colliding} words. Resolving...")
                
                used_sids = set(sid_tuples)
                
                for collision_sid, count in collisions.items():
                    collision_indices = [i for i, s in enumerate(sid_tuples) if s == collision_sid]
                    
                    # Keep first word, reassign others
                    for idx in collision_indices[1:]:
                        found = False
                        # Try all possible SID combinations
                        for l0 in range(codes_per_level):
                            if found:
                                break
                            for l1 in range(codes_per_level) if n_levels > 1 else [0]:
                                if found:
                                    break
                                for l2 in range(codes_per_level) if n_levels > 2 else [0]:
                                    if found:
                                        break
                                    for l3 in range(codes_per_level) if n_levels > 3 else [0]:
                                        new_sid = [l0, l1, l2, l3][:n_levels]
                                        if tuple(new_sid) not in used_sids:
                                            sids[idx] = new_sid
                                            used_sids.add(tuple(new_sid))
                                            sid_tuples[idx] = tuple(new_sid)
                                            found = True
                                            break
                
                # Verify uniqueness
                final_unique = len(set(tuple(s) for s in sids))
                if final_unique < vocab_size:
                    logger.error(f"Could not resolve all collisions: {final_unique}/{vocab_size} unique. "
                                f"Increase codes_per_level (need D^M >= V, currently {codes_per_level}^{n_levels}={codes_per_level**n_levels} < {vocab_size})")
                else:
                    logger.info(f"Collision resolution complete: {final_unique}/{vocab_size} unique SIDs (100%)")

        # Store SID info
        self._sid_vocab = vocab
        self._word_sids = sids
        self._sid_n_levels = n_levels
        self._sid_codes_per_level = codes_per_level
        self._sid_centroids = centroids_per_level  # Store centroids for F-ETM init

        # Compute Bag-of-Codes for all documents
        logger.info("Computing Bag-of-Codes for documents...")
        total_codes = n_levels * codes_per_level
        boc = np.zeros((len(self.texts), total_codes), dtype=np.float32)

        for level in range(n_levels):
            codes_at_level = sids[:, level]
            start_idx = level * codes_per_level

            for code in range(codes_per_level):
                mask = (codes_at_level == code).astype(np.float32)
                boc[:, start_idx + code] = (self.bow * mask).sum(axis=1)

        self.boc = boc

        # Log stats
        unique_sids = len(set(tuple(s) for s in sids))
        logger.info(f"Semantic IDs generated: {sids.shape}")
        logger.info(f"Unique SIDs: {unique_sids}/{vocab_size} ({100*unique_sids/vocab_size:.1f}%)")
        logger.info(f"Bag-of-Codes shape: {boc.shape}")
        logger.info(f"Compression: {vocab_size} words -> {total_codes} codes ({vocab_size / total_codes:.1f}x)")

        return sids, boc

    def inspect_semantic_ids(self, level: int = 0, code: int = 0, top_n: int = 20):
        """
        Inspect which words belong to a specific code at a given level.

        Parameters
        ----------
        level : int, optional
            Hierarchy level to inspect, by default 0.
        code : int, optional
            Code index at that level, by default 0.
        top_n : int, optional
            Number of words to return, by default 20.

        Returns
        -------
        list of str
            Words assigned to the specified code.
        """
        if not hasattr(self, "_word_sids"):
            raise ValueError("Must call get_semantic_ids() first")

        mask = self._word_sids[:, level] == code
        words = [self._sid_vocab[i] for i in np.where(mask)[0][:top_n]]
        return words

    def _compute_ppmi_svd_embeddings(self, n_components: int = 128, window: int = 5, random_state: int = 42):
        """
        Compute word embeddings from PPMI-weighted word co-occurrence matrix via SVD.
        
        This captures corpus-specific co-occurrence patterns that complement
        pretrained semantic embeddings.

        Parameters
        ----------
        n_components : int, optional
            Dimensionality of output embeddings, by default 128.
        window : int, optional
            Context window size for co-occurrence, by default 5.
        random_state : int, optional
            Random seed for SVD, by default 42.

        Returns
        -------
        np.ndarray
            Word embeddings of shape (vocab_size, n_components).
        """
        from sklearn.decomposition import TruncatedSVD
        from scipy.sparse import csr_matrix
        
        vocab = list(self._bow_vocab)
        word2idx = {w: i for i, w in enumerate(vocab)}
        vocab_size = len(vocab)
        
        logger.info(f"Building co-occurrence matrix (window={window})...")
        
        # Build co-occurrence matrix from tokenized documents
        cooc = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        for tokens in self.dataframe["tokens"]:
            token_ids = [word2idx.get(t) for t in tokens]
            token_ids = [t for t in token_ids if t is not None]
            
            for i, center_id in enumerate(token_ids):
                start = max(0, i - window)
                end = min(len(token_ids), i + window + 1)
                for j in range(start, end):
                    if i != j:
                        context_id = token_ids[j]
                        cooc[center_id, context_id] += 1
        
        # Compute PPMI
        logger.info("Computing PPMI...")
        total = cooc.sum()
        if total == 0:
            logger.warning("Empty co-occurrence matrix, falling back to random embeddings")
            np.random.seed(random_state)
            return np.random.randn(vocab_size, n_components).astype(np.float32)
        
        row_sums = cooc.sum(axis=1, keepdims=True)
        col_sums = cooc.sum(axis=0, keepdims=True)
        
        # PMI = log(P(w,c) / (P(w) * P(c))) = log(cooc * total / (row_sum * col_sum))
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log((cooc * total) / (row_sums * col_sums + 1e-10) + 1e-10)
        
        # PPMI: clip negative values
        ppmi = np.maximum(pmi, 0)
        
        # SVD
        logger.info(f"Running SVD (n_components={n_components})...")
        n_components = min(n_components, vocab_size - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        embeddings = svd.fit_transform(csr_matrix(ppmi))
        
        logger.info(f"PPMI-SVD explained variance: {svd.explained_variance_ratio_.sum():.2%}")
        
        return embeddings.astype(np.float32)
        return words
