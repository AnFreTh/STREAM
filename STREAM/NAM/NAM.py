import torch
from typing import List
import matplotlib.pyplot as plt
from typing import Tuple
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd


class FeatureNN(nn.Module):
    """
    Neural network model for predicting a single feature.

    Args:
        input_size (int, optional): Size of the input feature (default is 1).
        output_size (int, optional): Size of the output feature (default is 1).
        hidden_units (List[int], optional): List of hidden layer sizes (default is [64, 32]).
        activation (nn.Module, optional): Activation function to use (default is nn.ReLU()).
        dropout (float, optional): Dropout probability (default is 0.3).

    Attributes:
        model (nn.Sequential): Sequential neural network model.

    Methods:
        forward(x): Forward pass of the neural network.
        plot(feature_name, ax=None): Plot the model's predictions for a specific feature.
        plot_data(x, y, feature_name, ax=None): Plot the model's predictions and actual data points for a specific feature.
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        hidden_units: List[int] = None,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.5,
    ):
        super().__init__()
        hidden_units = hidden_units or [128, 64, 32]  # Default hidden units

        layers = [nn.Linear(input_size, hidden_units[0]), activation]
        for i in range(1, len(hidden_units)):
            layers.extend(
                [
                    nn.Linear(hidden_units[i - 1], hidden_units[i]),
                    activation,
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_units[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def plot(self, feature_name, ax=None):
        """
        Plot the model's predictions for a specific feature.

        Args:
            feature_name (str): Name of the feature.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on (default is None).

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        with torch.no_grad():
            x_axis = torch.linspace(-1, 1, 500).unsqueeze(1)  # Ensure input is 2D

            # Plot the model's predictions for the specific feature
            ax.plot(
                x_axis.numpy(),
                self.forward(x_axis).numpy(),
                linestyle="solid",
                linewidth=1,
                color="red",
            )

            # Set the y-axis label to the feature name
            ax.set_ylabel(feature_name)

    def plot_data(self, x, y, feature_name, ax=None):
        """
        Plot the model's predictions and actual data points for a specific feature.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Actual target data.
            feature_name (str): Name of the feature.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on (default is None).

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        with torch.no_grad():
            x_axis = torch.linspace(-1, 1, 500).unsqueeze(1)

            # Plot the model's predictions and actual data points for the specific feature
            ax.plot(
                x_axis.numpy(),
                self.forward(x_axis).numpy(),
                linestyle="solid",
                linewidth=1,
                color="red",
            )
            ax.scatter(x, y, color="gray", s=2, alpha=0.3)

            # Set the y-axis label to the feature name
            ax.set_ylabel(feature_name)


class NeuralAdditiveModel(nn.Module):
    """
    Neural Additive Model for combining multiple feature-specific neural networks.

    Args:
        input_size (int): Size of the input feature space.
        output_size (int): Size of the output feature space.
        hidden_units (List[int], optional): List of hidden layer sizes for feature-specific neural networks (default is [64, 32]).
        feature_dropout (float, optional): Dropout probability for input features (default is 0.0).
        hidden_dropout (float, optional): Dropout probability for hidden layers (default is 0.3).
        activation (str, optional): Activation function for hidden layers (default is "relu").
        out_activation (nn.Module, optional): Activation function for output layer (default is None).

    Attributes:
        input_size (int): Size of the input feature space.
        hidden_units (List[int]): List of hidden layer sizes for feature-specific neural networks.
        feature_dropout (nn.Dropout): Dropout layer for input features.
        bias (nn.Parameter): Bias parameter for the output layer.
        out_activation (nn.Module): Activation function for the output layer.
        feature_nns (nn.ModuleList): List of feature-specific neural networks.

    Methods:
        forward(x): Forward pass of the neural network.
        plot(): Plot the learned functions for each feature-specific neural network.
        plot_data(x, y): Plot the learned functions and actual data points for each feature-specific neural network.
        _get_activation_fn(activation): Get the activation function based on the provided string.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: List[int] = None,
        feature_dropout: float = 0.0,
        hidden_dropout: float = 0.3,
        activation: str = "relu",
        out_activation=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units or [64, 32]  # Default hidden units
        self.feature_dropout = nn.Dropout(p=feature_dropout)
        self.bias = nn.Parameter(torch.zeros(output_size))

        # Set up the activation function based on the string
        activation_fn = self._get_activation_fn(activation)

        self.out_activation = (
            out_activation if out_activation is not None else nn.Identity()
        )

        # Create feature-specific networks using FeatureNN
        self.feature_nns = nn.ModuleList(
            [
                FeatureNN(
                    input_size=1,  # Each feature-specific NN takes a single feature as input
                    output_size=output_size,
                    hidden_units=hidden_units,
                    activation=activation_fn,
                    dropout=hidden_dropout,
                )
                for _ in range(input_size)
            ]
        )

    def _get_activation_fn(self, activation):
        """
        Get the activation function based on the provided string.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.Module: Activation function module.
        """
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        feature_outputs = [nn(x[:, i : i + 1]) for i, nn in enumerate(self.feature_nns)]
        output = torch.cat(feature_outputs, dim=1).sum(dim=1, keepdim=True) + self.bias
        return self.out_activation(output)

    def plot(self):
        """
        Plot the learned functions for each feature-specific neural network.

        Returns:
            None
        """
        self.eval()
        with torch.no_grad():
            if len(self.feature_nns) > 1:
                fig, axes = plt.subplots(len(self.feature_nns), 1, figsize=(10, 7))
                for i, ax in enumerate(axes.flat):
                    component = self.feature_nns[i]
                    component.plot(ax)
            else:
                self.feature_nns[0].plot()

    def plot_data(self, x, y):
        """
        Plot the learned functions and actual data points for each feature-specific neural network.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Actual target data.

        Returns:
            None
        """
        self.eval()
        with torch.no_grad():
            if len(self.feature_nns) > 1:
                fig, axes = plt.subplots(len(self.feature_nns), 1, figsize=(10, 7))
                for i, ax in enumerate(axes.flat):
                    component = self.feature_nns[i]
                    component.plot_data(ax, x[i], y)
            else:
                self.feature_nns[0].plot()


class DownstreamModel(pl.LightningModule):
    """
    PyTorch Lightning module for downstream modeling using a trained topic model.

    Args:
        trained_topic_model (AbstractModel): Trained topic model.
        target_column (str): Name of the target column.
        dataset (AbstractDataset, optional): Dataset object (default is None).
        structured_data (pd.DataFrame, optional): Structured data (default is None).
        task (str, optional): Type of task, either 'regression' or 'classification' (default is 'regression').
        batch_size (int, optional): Batch size for training (default is 128).
        lr (float, optional): Learning rate for optimization (default is 0.0005).
        hidden_units (List[int], optional): List of hidden layer sizes for the Neural Additive Model (default is None).
        feature_dropout (float, optional): Dropout probability for input features (default is 0.0).
        hidden_dropout (float, optional): Dropout probability for hidden layers (default is 0.3).
        activation (str, optional): Activation function for hidden layers (default is 'relu').
        out_activation (nn.Module, optional): Activation function for output layer (default is None).

    Attributes:
        trained_topic_model (AbstractModel): Trained topic model.
        task (str): Type of task, either 'regression' or 'classification'.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimization.
        loss_fn (nn.Module): Loss function for the task.
        structured_data (pd.DataFrame): Structured data used for downstream modeling.
        target_column (str): Name of the target column.
        combined_data (pd.DataFrame): Combined DataFrame containing structured data and topic probabilities.
        model (NeuralAdditiveModel): Neural Additive Model for downstream modeling.

    Methods:
        prepare_combined_data(): Prepare combined DataFrame containing structured data and topic probabilities.
        preprocess_structured_data(data): Preprocess structured data.
        define_nam_model(hidden_units, feature_dropout, hidden_dropout, activation, out_activation): Define the Neural Additive Model architecture.
        forward(x): Forward pass of the model.
        training_step(batch, batch_idx): Training step for LightningModule.
        validation_step(batch, batch_idx): Validation step for LightningModule.
        test_step(batch, batch_idx): Test step for LightningModule.
        configure_optimizers(): Configure optimizer for training.
        setup(stage): Setup datasets for training and validation.
        train_dataloader(): DataLoader for training dataset.
        val_dataloader(): DataLoader for validation dataset.
        get_feature_names(): Get names of input features.
        plot_feature_nns(): Plot the learned functions for each feature-specific neural network.
    """

    def __init__(
        self,
        trained_topic_model,
        target_column,
        dataset=None,
        structured_data=None,
        task="regression",
        batch_size=128,
        lr=0.0005,
        hidden_units: List[int] = None,
        feature_dropout: float = 0.0,
        hidden_dropout: float = 0.3,
        activation: str = "relu",
        out_activation=None,
    ):
        super().__init__()
        self.trained_topic_model = trained_topic_model
        self.task = task
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

        if dataset is None:
            self.structured_data = (
                self.trained_topic_model.dataset.get_structured_data()
            )
        else:
            self.structured_data = dataset.get_structured_data(data=structured_data)
        self.target_column = target_column

        # Combine topic probabilities with structured data
        self.combined_data = self.prepare_combined_data()

        # Define the NAM architecture here based on the shape of the combined data
        self.model = self.define_nam_model(
            hidden_units=hidden_units,
            feature_dropout=feature_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            out_activation=out_activation,
        )

    def prepare_combined_data(self):
        """
        Prepare combined DataFrame containing structured data and topic probabilities.

        Returns:
            pd.DataFrame: Combined DataFrame.
        """
        # Preprocess structured data
        preprocessed_structured_data = self.preprocess_structured_data(
            self.structured_data
        )

        # Assert that trained_topic_model has the _get_topic_document_matrix method
        assert hasattr(
            self.trained_topic_model, "_get_topic_document_matrix"
        ), "trained_topic_model must have a method called _get_topic_document_matrix."

        # Use _get_topic_document_matrix function to get the topic-document matrix
        topic_document_matrix = self.trained_topic_model._get_topic_document_matrix()

        # Convert the matrix to a DataFrame and transpose it to shape (n, k)
        topic_probabilities = pd.DataFrame(topic_document_matrix).transpose()
        new_column_names = [f"Topic_{i}" for i in range(topic_probabilities.shape[1])]
        topic_probabilities.columns = new_column_names

        preprocessed_structured_data = preprocessed_structured_data.reset_index(
            drop=True
        )
        topic_probabilities = topic_probabilities.reset_index(drop=True)

        # Combine the preprocessed structured data with the topic probabilities
        combined_df = pd.concat(
            [preprocessed_structured_data, topic_probabilities], axis=1
        )

        # Ensure the target column is the last column in the DataFrame
        combined_df = combined_df[
            [col for col in combined_df.columns if col != self.target_column]
            + [self.target_column]
        ]

        combined_df = combined_df.dropna()

        return combined_df

    def preprocess_structured_data(self, data):
        """
        Preprocess structured data.

        Args:
            data (pd.DataFrame): Structured data.

        Returns:
            pd.DataFrame: Preprocessed structured data.
        """
        # Identify categorical and numerical columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns

        transformers = []

        if len(numerical_cols) > 0:
            # Define preprocessing for numerical columns (scale them)
            numerical_transformer = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="mean"),
                    ),  # Impute missing values with mean
                    ("scaler", StandardScaler()),  # Scale numerical variables
                ]
            )
            transformers.append(("num", numerical_transformer, numerical_cols))

        if len(categorical_cols) > 0:
            # Define preprocessing for categorical columns (one-hot encode them)
            categorical_transformer = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # Impute missing values with the most frequent value
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),  # One-hot encode categorical variables
                ]
            )
            transformers.append(("cat", categorical_transformer, categorical_cols))

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(transformers=transformers)

        # Fit and transform the data
        preprocessed_data = preprocessor.fit_transform(data)

        # Get the feature names for numerical columns
        feature_names = numerical_cols.tolist()

        # Extend feature names with those generated by one-hot encoding for categorical columns, if any
        if len(categorical_cols) > 0:
            feature_names += list(
                preprocessor.named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names_out(categorical_cols)
            )

        # Convert the transformed data back to a DataFrame
        preprocessed_data = pd.DataFrame(
            preprocessed_data, columns=feature_names, index=data.index
        )

        return preprocessed_data

    def define_nam_model(
        self, hidden_units, feature_dropout, hidden_dropout, activation, out_activation
    ):
        """
        Define the Neural Additive Model architecture.

        Args:
            hidden_units (List[int]): List of hidden layer sizes for the Neural Additive Model.
            feature_dropout (float): Dropout probability for input features.
            hidden_dropout (float): Dropout probability for hidden layers.
            activation (str): Activation function for hidden layers.
            out_activation (nn.Module): Activation function for output layer.

        Returns:
            NeuralAdditiveModel: Initialized Neural Additive Model.
        """
        input_size = self.combined_data.shape[1] - 1  # Exclude target column
        output_size = (
            1
            if self.task == "regression"
            else len(self.combined_data[self.target_column].unique())
        )

        model = NeuralAdditiveModel(
            input_size=input_size,
            output_size=output_size,
            hidden_units=hidden_units,
            feature_dropout=feature_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            out_activation=out_activation,
        )
        return model

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        """
        Training step for LightningModule.

        Args:
            batch (Tuple): Batch of input data and target.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for LightningModule.

        Args:
            batch (Tuple): Batch of input data and target.
            batch_idx (int): Batch index.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        """
        Test step for LightningModule.

        Args:
            batch (Tuple): Batch of input data and target.
            batch_idx (int): Batch index.
        """
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y)
        self.log(
            "test_loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        """
        Configure optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.lr)

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.
        """
        # Split the combined data into features and target
        X = self.combined_data.iloc[:, :-1].values  # Exclude target column
        y = self.combined_data.iloc[:, -1].values

        # Convert to PyTorch tensors
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )

        # Train-validation split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        """
        DataLoader for training dataset.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        DataLoader for validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def get_feature_names(self):
        """
        Get names of input features.

        Returns:
            List[str]: List of feature names.
        """
        # Assuming the last column of combined_data is the target, and all other columns are features
        return self.combined_data.columns[:-1].tolist()

    def plot_feature_nns(self):
        """
        Plot the learned functions for each feature-specific neural network.
        """
        feature_names = self.get_feature_names()  # Retrieve feature names
        num_features = len(self.model.feature_nns)

        fig, axs = plt.subplots(num_features, 1, figsize=(10, num_features * 2))

        for i, feature_nn in enumerate(self.model.feature_nns):
            ax = axs[i] if num_features > 1 else axs
            feature_nn.plot(
                feature_names[i], ax=ax
            )  # Pass the feature name to the plot method
            ax.set_title(f"Feature: {feature_names[i]}")

        plt.tight_layout()
        plt.show()
