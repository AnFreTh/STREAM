import importlib.util
import os
import pickle
from urllib.parse import urljoin

import pandas as pd
import requests
from loguru import logger

PACKAGE_NAME = "stream_topic"


class DataDownloader:

    def __init__(self, name=None, language="en"):

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
            "detokenize": False,
        }

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
        # Get the location of the installed package
        spec = importlib.util.find_spec(PACKAGE_NAME)
        package_root_dir = os.path.dirname(spec.origin)
        if package_root_dir is None:
            raise ImportError(f"Cannot find the package '{PACKAGE_NAME}'")

        # Construct the full path to the dataset
        dataset_path = os.path.join(
            package_root_dir, "stream_topic_data/preprocessed_datasets", name
        )

        return dataset_path

    def has_embeddings(
        self, embedding_model_name, path=None, file_name=None, source="github"
    ):
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
        if source == "github" and path is None:
            BASE_URL = "https://raw.githubusercontent.com/mkumar73/stream_topic_data/main/datasets/pre_embedded_datasets/"
            git_pkl_path = urljoin(
                BASE_URL,
                os.path.join(
                    self.name, f"{self.name}_embeddings_{embedding_model_name}.pkl"
                ).replace(os.sep, "/"),
            )
            return url_exists(git_pkl_path)

        elif path is None:
            path = self.get_package_embeddings_path(self.name)
            embeddings_file = (
                urljoin(path, file_name)
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
        try:
            if path is None:
                path = self.get_package_embeddings_path(self.name)

            logger.info(f"Saving embeddings to path: {path}")

            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"Created directory: {path}")

            embeddings_file = (
                os.path.join(path, file_name)
                if file_name
                else os.path.join(
                    path, f"{self.name}_embeddings_{embedding_model_name}.pkl"
                )
            )

            logger.info(f"Embeddings file path: {embeddings_file}")

            with open(embeddings_file, "wb") as file:
                pickle.dump(embeddings, file)

            logger.info("Embeddings saved successfully.")

        except PermissionError as e:
            logger.error(f"PermissionError: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def get_embeddings(
        self, embedding_model_name, path=None, file_name=None, source="github"
    ):
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
        if source == "github" and path is None:
            # logger.info(f"Fetching dataset from github")
            self.load_custom_dataset_from_url(
                self.name, embeddings=True, embedding_model_name=embedding_model_name
            )

        elif not self.has_embeddings(embedding_model_name, path, file_name):
            raise ValueError(
                "Embeddings are not available. Run the encoding process first or load embeddings."
            )

        # logger.info("--- Loading pre-computed document embeddings ---")

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
        # Get the location of the installed package
        spec = importlib.util.find_spec(PACKAGE_NAME)
        package_root_dir = os.path.dirname(spec.origin)
        if package_root_dir is None:
            raise ImportError(f"Cannot find the package '{PACKAGE_NAME}'")

        # Construct the full path to the dataset
        embedding_path = os.path.join(
            package_root_dir, "stream_topic_data", "pre_embedded_datasets", name
        )

        return embedding_path

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

            self.dataframe["tokens"] = self.dataframe["text"].apply(
                lambda x: x.split())
            self.texts = self.dataframe["text"].tolist()
            self.labels = self.dataframe["labels"].tolist()

    def load_custom_dataset_from_url(
        self, dataset_path=None, embeddings=False, embedding_model_name=None
    ):
        """
        Load a custom dataset from a folder.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset folder.
        """
        if embeddings:
            if not embedding_model_name:
                raise ValueError(
                    "Please provide the embedding model name to load embeddings."
                )
            BASE_URL = "https://raw.githubusercontent.com/mkumar73/stream_topic_data/main/datasets/pre_embedded_datasets/"
            git_pkl_path = urljoin(
                BASE_URL,
                os.path.join(
                    self.name, f"{self.name}_embeddings_{embedding_model_name}.pkl"
                ).replace(os.sep, "/"),
            )
            data_home = get_data_home()
            save_dir = os.path.join(
                data_home, "pre_embedded_datasets", self.name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            local_pkl_path = os.path.join(
                save_dir, f"{self.name}_embeddings_{embedding_model_name}.pkl"
            )

            if url_exists(git_pkl_path):
                logger.info(f"Downloading embeddings from github")
                download_file_from_github(git_pkl_path, local_pkl_path)
                logger.info(
                    f"Embeddings  downloaded successfully at ~/stream_topic_data/"
                )

        else:
            BASE_URL = "https://raw.githubusercontent.com/mkumar73/stream_topic_data/main/datasets/preprocessed_datasets/"
            git_parquet_path = urljoin(
                BASE_URL,
                os.path.join(self.name, f"{self.name}.parquet").replace(
                    os.sep, "/"),
            )
            git_pkl_path = urljoin(
                BASE_URL,
                os.path.join(self.name, f"{self.name}_info.pkl").replace(
                    os.sep, "/"),
            )

            data_home = get_data_home()
            save_dir = os.path.join(
                data_home, "preprocessed_datasets", self.name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            local_parquet_path = os.path.join(save_dir, f"{self.name}.parquet")
            local_pkl_path = os.path.join(save_dir, f"{self.name}_info.pkl")

            if url_exists(git_parquet_path):
                logger.info(f"Downloading dataset from github")
                download_file_from_github(git_parquet_path, local_parquet_path)
                logger.info(
                    f"Dataset downloaded successfully at ~/stream_topic_data/")
                self.load_dataset_from_parquet(local_parquet_path)
            else:
                # TODO: need to be refactored to include githb url for corpus and labels
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

                self.dataframe["tokens"] = self.dataframe["text"].apply(
                    lambda x: x.split()
                )
                self.texts = self.dataframe["text"].tolist()
                self.labels = self.dataframe["labels"].tolist()

            if url_exists(git_pkl_path):
                logger.info(f"Downloading dataset info from github")
                download_file_from_github(git_pkl_path, local_pkl_path)
                logger.info(
                    f"Dataset info downloaded successfully at ~/stream_topic_data/"
                )

    def save_word_embeddings(
        self, word_embeddings, model_name, path=None, file_name=None
    ):
        """
        Save word embeddings for the dataset.

        Parameters
        ----------
        word_embeddings : dict
            Word embeddings to save.
        model_name : str
            Name of the pre-trained model.
        """
        self.save_embeddings(
            embeddings=word_embeddings,
            embedding_model_name=model_name,
            path=path,
            file_name=file_name,
        )

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
        self.dataframe["tokens"] = self.dataframe["text"].apply(
            lambda x: x.split())
        self.texts = self.dataframe["text"].tolist()
        self.labels = self.dataframe["labels"].tolist()


def get_data_home(data_home=None):
    """
    Get the data home directory.

    Parameters
    ----------
    data_home : str, optional
        Path to the data home directory, defaults to None.


    Notes
    -----
    If environment variable STREAM_TOPIC_DATA is not set, the default path is `~/stream_topic_data`.

    Returns
    -------
    str
        Path to the data home directory.

    """
    spec = importlib.util.find_spec(PACKAGE_NAME)
    package_root_dir = os.path.dirname(spec.origin)
    if data_home is None:
        data_home = os.environ.get(
            "STREAM_TOPIC_DATA", os.path.join(
                package_root_dir, "stream_topic_data")
        )
    # data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home


def url_exists(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False


def download_file_from_github(url: str, save_dir: str):
    """
    Downloads a file from a GitHub repository.

    Parameters
    ----------
    url : str
        URL of the file.
    save_path : str
        Path to save the file.
    """
    response = requests.get(url)
    response.raise_for_status()  # Check if the download was successful

    with open(save_dir, "wb") as file:
        file.write(response.content)
    # logger.info(f"File downloaded and saved to {save_dir}")
