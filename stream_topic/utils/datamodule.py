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
            b, self.vocab = dataset.get_bow(**kwargs)
        if tf_idf:
            tfidf, self.vocab = dataset.get_tfidf(**kwargs)
        if word_embeddings:
            self.wembs = dataset.get_word_embeddings(**kwargs)

        self.train_dataset, self.val_dataset = dataset.split_dataset(
            train_ratio=train, val_ratio=val, seed=random_state
        )

    def train_dataloader(self):
        """
        Returns the training dataloader.

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
            **self.dataloader_kwargs,
        )
