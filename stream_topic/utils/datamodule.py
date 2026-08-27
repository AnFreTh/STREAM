import lightning as pl
from torch.utils.data import DataLoader


class TMDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning data module for managing training and validation data loaders in a structured way.

    This class simplifies the process of batch-wise data loading for training and validation datasets during
    the training loop, and is particularly useful when working with PyTorch Lightning's training framework.

    Parameters:
        preprocessor: object
            An instance of your preprocessor class.
        batch_size: int
            Size of batches for the DataLoader.
        shuffle: bool
            Whether to shuffle the training data in the DataLoader.
        X_val: DataFrame or None, optional
            Validation features. If None, uses train-test split.
        y_val: array-like or None, optional
            Validation labels. If None, uses train-test split.
        val_size: float, optional
            Proportion of data to include in the validation split if `X_val` and `y_val` are None.
        random_state: int, optional
            Random seed for reproducibility in data splitting.
        regression: bool, optional
            Whether the problem is regression (True) or classification (False).
    """

    def __init__(
        self,
        batch_size,
        shuffle,
        val_size=0.2,
        random_state=101,
        **dataloader_kwargs,
    ):
        """
        Initialize the data module with the specified preprocessor, batch size, shuffle option,
        and optional validation data settings.

        Args:
            preprocessor (object): An instance of the preprocessor class for data preprocessing.
            batch_size (int): Size of batches for the DataLoader.
            shuffle (bool): Whether to shuffle the training data in the DataLoader.
            X_val (DataFrame or None, optional): Validation features. If None, uses train-test split.
            y_val (array-like or None, optional): Validation labels. If None, uses train-test split.
            val_size (float, optional): Proportion of data to include in the validation split if `X_val` and `y_val` are None.
            random_state (int, optional): Random seed for reproducibility in data splitting.
            regression (bool, optional): Whether the problem is regression (True) or classification (False).
        """
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cat_feature_info = None
        self.num_feature_info = None
        self.val_size = val_size
        self.random_state = random_state

        # Initialize placeholders for data
        self.X_train = None
        self.X_val = None
        self.dataloader_kwargs = dataloader_kwargs

    def preprocess_data(
        self,
        dataset,
        train=0.8,
        val=0.2,
        embeddings=False,
        bow=False,
        tf_idf=False,
        word_embeddings=False,
        tokens=False,
        random_state=101,
        embedding_model_name=None,
        **kwargs,
    ):

        if embeddings:
            if dataset.embeddings is None:
                embs = dataset.get_embeddings(embedding_model_name)
            else:
                embs = dataset.embeddings

            dataset.embeddings = embs

        if bow:
            if dataset.bow is None:
                # Only create BOW if it doesn't exist yet
                b, self.vocab = dataset.get_bow(**kwargs)
            else:
                # Use existing preprocessed BOW
                if hasattr(dataset, "_bow_vocab"):
                    self.vocab = dataset._bow_vocab
                else:
                    # Fallback: get vocab from existing BOW
                    _, self.vocab = dataset.get_bow()
        if tf_idf:
            if dataset.tfidf is None:
                # Only create TF-IDF if it doesn't exist yet
                tfidf, self.vocab = dataset.get_tfidf(**kwargs)
            else:
                # Use existing preprocessed TF-IDF
                if hasattr(dataset, "_tfidf_vocab"):
                    self.vocab = dataset._tfidf_vocab
                else:
                    # Fallback: get vocab from existing TF-IDF
                    _, self.vocab = dataset.get_tfidf()
        if word_embeddings:
            self.wembs = dataset.get_word_embeddings(**kwargs)
        if tokens:
            dataset.tokens = self._prepare_tokens(
                dataset, self.vocab if bow else None, **kwargs
            )

        self.train_dataset, self.val_dataset = dataset.split_dataset(
            train_ratio=train, val_ratio=val, seed=random_state
        )

    def _prepare_tokens(self, dataset, vocab=None, **kwargs):
        """Prepare token indices for models that use sequences."""
        import numpy as np

        if vocab is None:
            _, vocab = dataset.get_bow(**kwargs)

        vocab_list = vocab.tolist()
        word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}

        tokens_list = []
        for text in dataset.texts:
            words = text.split()
            token_ids = [word_to_idx.get(w, 0) for w in words]  # 0 for unknown
            tokens_list.append(np.array(token_ids, dtype=np.int64))

        return tokens_list

    def train_dataloader(self):
        """
        Returns the training dataloader.

        Returns:
            DataLoader: DataLoader instance for the training dataset.
        """
        # Only drop the trailing batch when it would contain EXACTLY one sample
        # (BatchNorm's "Expected more than 1 value per channel" crash). A broader
        # guard (e.g. drop_last=True whenever n>batch_size) would drop up to
        # batch_size-1 documents on small datasets under HPO with large batch
        # sizes -- meaningful data loss per epoch. Full-batch models (FASTopic)
        # keep their single batch: n <= batch_size -> n % batch_size == n != 1.
        n = len(self.train_dataset)
        drop_last = (n > self.batch_size) and (n % self.batch_size == 1)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate_fn,
            drop_last=drop_last,
            **self.dataloader_kwargs,
        )

    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length sequences."""
        import torch
        import numpy as np

        collated = {}

        # Handle each key in the batch
        for key in batch[0].keys():
            if key == "tokens":
                # Pad token sequences
                tokens_list = [item[key] for item in batch]
                max_len = max(len(t) for t in tokens_list)
                padded = np.zeros((len(tokens_list), max_len), dtype=np.int64)
                for i, tokens in enumerate(tokens_list):
                    padded[i, : len(tokens)] = tokens
                collated[key] = torch.from_numpy(padded)
            elif key == "text":
                collated[key] = [item[key] for item in batch]
            elif key in ["bow", "tfidf", "embedding", "features", "boc"]:
                collated[key] = torch.stack(
                    [
                        (
                            torch.from_numpy(item[key])
                            if isinstance(item[key], np.ndarray)
                            else item[key]
                        )
                        for item in batch
                    ]
                )
            else:
                # Default: try to stack
                try:
                    collated[key] = torch.stack(
                        [
                            (
                                item[key]
                                if torch.is_tensor(item[key])
                                else torch.tensor(item[key])
                            )
                            for item in batch
                        ]
                    )
                except:
                    collated[key] = [item[key] for item in batch]

        return collated

    def predict_dataloader(self):
        """
        Returns the predict dataloader for the complete training dataset.

        Returns:
            DataLoader: DataLoader instance for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: DataLoader instance for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            **self.dataloader_kwargs,
        )
