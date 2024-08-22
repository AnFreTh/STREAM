import re
import unicodedata
from collections import Counter
from typing import List, Set

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm


class TextPreprocessor:
    """
    Text preprocessor class for cleaning and preprocessing text data.

    Parameters
    ----------
    language : str, optional
        Language of the text data (default is "en").
    remove_stopwords : bool, optional
        Whether to remove stopwords from the text data (default is False).
    lowercase : bool, optional
        Whether to convert text to lowercase (default is True).
    remove_punctuation : bool, optional
        Whether to remove punctuation from the text data (default is True).
    remove_numbers : bool, optional
        Whether to remove numbers from the text data (default is True).
    lemmatize : bool, optional
        Whether to lemmatize words in the text data (default is False).
    stem : bool, optional
        Whether to stem words in the text data (default is False).
    expand_contractions : bool, optional
        Whether to expand contractions in the text data (default is True).
    remove_html_tags : bool, optional
        Whether to remove HTML tags from the text data (default is True).
    remove_special_chars : bool, optional
        Whether to remove special characters from the text data (default is True).
    remove_accents : bool, optional
        Whether to remove accents from the text data (default is True).
    custom_stopwords : set, optional
        Custom stopwords to remove from the text data (default is []).
    detokenize : bool, optional
        Whether to detokenize the text data (default is False).
    min_word_freq : int, optional
        Minimum word frequency to keep in the text data (default is 2).
    max_word_freq : int, optional
        Maximum word frequency to keep in the text data (default is None).
    min_word_length : int, optional
        Minimum word length to keep in the text data (default is 3).
    max_word_length : int, optional
        Maximum word length to keep in the text data (default is None).
    dictionary : set, optional
        Dictionary of words to keep in the text data (default is []).
    remove_words_with_numbers : bool, optional
        Whether to remove words containing numbers from the text data (default is False).
    remove_words_with_special_chars : bool, optional
        Whether to remove words containing special characters from the text data (default is False).

    """

    def __init__(self, **kwargs):
        self.language = kwargs.get("language", "en")
        self.remove_stopwords = kwargs.get("remove_stopwords", False)
        self.lowercase = kwargs.get("lowercase", True)
        self.remove_punctuation = kwargs.get("remove_punctuation", True)
        self.remove_numbers = kwargs.get("remove_numbers", True)
        self.lemmatize = kwargs.get("lemmatize", False)
        self.stem = kwargs.get("stem", False)
        self.expand_contractions = kwargs.get("expand_contractions", True)
        self.remove_html_tags = kwargs.get("remove_html_tags", True)
        self.remove_special_chars = kwargs.get("remove_special_chars", True)
        self.remove_accents = kwargs.get("remove_accents", True)
        self.custom_stopwords = (
            set(kwargs.get("custom_stopwords", []))
            if kwargs.get("custom_stopwords")
            else set()
        )
        self.detokenize = kwargs.get("detokenize", False)
        self.min_word_freq = kwargs.get("min_word_freq", 2)
        self.max_word_freq = kwargs.get("max_word_freq", None)
        self.min_word_length = kwargs.get("min_word_length", 3)
        self.max_word_length = kwargs.get("max_word_length", None)
        self.dictionary = set(kwargs.get("dictionary", []))
        self.remove_words_with_numbers = kwargs.get("remove_words_with_numbers", False)
        self.remove_words_with_special_chars = kwargs.get(
            "remove_words_with_special_chars", False
        )

        if self.language != "en" and self.remove_stopwords:
            self.stop_words = set(stopwords.words(self.language))
        else:
            self.stop_words = set(stopwords.words("english"))

        self.stop_words.update(self.custom_stopwords)

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

        if self.stem:
            self.stemmer = PorterStemmer()

        self.contractions_dict = self._load_contractions()
        self.word_freq = Counter()

    def _load_contractions(self):
        # Load a dictionary of contractions and their expansions
        contractions_dict = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am",
        }
        return contractions_dict

    def _expand_contractions(self, text):
        contractions_pattern = re.compile(
            "({})".format("|".join(self.contractions_dict.keys())),
            flags=re.IGNORECASE | re.DOTALL,
        )

        def expand_match(contraction):
            match = contraction.group(0)
            expanded_contraction = self.contractions_dict.get(match.lower())
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        return expanded_text

    def _remove_html_tags(self, text):
        clean = re.compile("<.*?>")
        return re.sub(clean, " ", text)

    def _remove_special_characters(self, text):
        return re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    def _remove_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore")
        return text.decode("utf-8")

    def _clean_text(self, text):
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        if self.expand_contractions:
            text = self._expand_contractions(text)
        if self.remove_html_tags:
            text = self._remove_html_tags(text)
        if self.remove_special_chars:
            text = self._remove_special_characters(text)
        if self.remove_accents:
            text = self._remove_accents(text)
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        words = word_tokenize(text)

        # Update word frequency counter
        self.word_freq.update(words)

        if self.remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        if self.lemmatize:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        if self.stem:
            words = [self.stemmer.stem(word) for word in words]

        if self.min_word_freq is not None:
            words = [
                word for word in words if self.word_freq[word] >= self.min_word_freq
            ]

        if self.max_word_freq is not None:
            words = [
                word for word in words if self.word_freq[word] <= self.max_word_freq
            ]

        if self.min_word_length is not None:
            words = [word for word in words if len(word) >= self.min_word_length]

        if self.max_word_length is not None:
            words = [word for word in words if len(word) <= self.max_word_length]

        if self.dictionary != set():
            words = [word for word in words if word in self.dictionary]

        if self.remove_words_with_numbers:
            words = [word for word in words if not any(char.isdigit() for char in word)]

        if self.remove_words_with_special_chars:
            words = [word for word in words if not re.search(r"[^a-zA-Z0-9\s]", word)]

        if self.detokenize:
            text = TreebankWordDetokenizer().detokenize(words)
        else:
            text = " ".join(words)

        # Remove double spaces
        text = re.sub(r"\s+", " ", text)

        return text

    def preprocess_text(self, text):
        """
        Preprocess a single text document.

        Parameters
        ----------
        text : str
            Text document to preprocess.

        Returns
        -------
        str
            Preprocessed text document.

        """
        try:
            language = detect(text)
            if language != self.language:
                return text
        except LangDetectException:
            pass
        return self._clean_text(text)

    def preprocess_dataframe(self, df, text_column):
        """
        Preprocess a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing text data.
        text_column : str
            Name of the column containing text data.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.

        """
        df[text_column] = df[text_column].apply(self.preprocess_text)
        return df

    def preprocess_documents(self, documents: List[str]) -> List[str]:
        preprocessed_docs = []
        for doc in tqdm(documents, desc="Preprocessing documents"):
            preprocessed_docs.append(self.preprocess_text(doc))
        return preprocessed_docs

    def add_custom_stopwords(self, stopwords: Set[str]):
        """
        Add custom stopwords to the preprocessor.

        Parameters
        ----------
        stopwords : set
            Set of custom stopwords to be added.
        """
        self.custom_stopwords.update(stopwords)
        self.stop_words.update(stopwords)

    def remove_custom_stopwords(self, stopwords: Set[str]):
        """
        Remove custom stopwords from the preprocessor.

        Parameters
        ----------
        stopwords : set
            Set of custom stopwords to be removed.
        """
        self.custom_stopwords.difference_update(stopwords)
        self.stop_words.difference_update(stopwords)
