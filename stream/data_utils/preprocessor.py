import re
import unicodedata
from typing import List, Set
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
from tqdm import tqdm

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class TextPreprocessor:
    def __init__(self, **kwargs):
        self.language = kwargs.get("language", "english")
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
        self.custom_stopwords = set(kwargs.get("custom_stopwords", []))
        self.detokenize = kwargs.get("detokenize", True)

        if self.language != "english" and self.remove_stopwords:
            self.stop_words = set(stopwords.words(self.language))
        else:
            self.stop_words = set(stopwords.words("english"))

        self.stop_words.update(self.custom_stopwords)

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

        if self.stem:
            self.stemmer = PorterStemmer()

        self.contractions_dict = self._load_contractions()

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
        return re.sub(clean, "", text)

    def _remove_special_characters(self, text):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

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
            text = re.sub(r"\d+", "", text)
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        words = word_tokenize(text)

        if self.remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        if self.lemmatize:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        if self.stem:
            words = [self.stemmer.stem(word) for word in words]

        if self.detokenize:
            text = TreebankWordDetokenizer().detokenize(words)
        else:
            text = " ".join(words)

        return text

    def preprocess_text(self, text):
        try:
            language = detect(text)
            if language != self.language:
                return text
        except LangDetectException:
            pass
        return self._clean_text(text)

    def preprocess_dataframe(self, df, text_column):
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
