# In stream_topic/preprocessor/arabic_preprocessing.py

import os
import re
import unicodedata
import nltk
from pathlib import Path
from typing import List

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except Exception as e:
    print(f"NLTK download failed: {e}")
    NLTK_AVAILABLE = False

try:
    from pyarabic import araby
    from pyarabic.normalize import normalize_hamza, normalize_lamalef
    PYARABIC_AVAILABLE = True
except ImportError:
    print("PyArabic not available. Installing basic preprocessing...")
    PYARABIC_AVAILABLE = False

class ArabicPreprocessor:
    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        remove_stopwords: bool = True,
        normalize_arabic: bool = True
    ):
        self.remove_diacritics = remove_diacritics
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.normalize_arabic = normalize_arabic
        
        # Basic Arabic stopwords
        self.arabic_stopwords = {
            'في', 'من', 'إلى', 'على', 'أو', 'ثم', 'حتى', 'إذا', 'نحو',
            'لدى', 'عند', 'أن', 'التي', 'الذي', 'هذا', 'هذه', 'هؤلاء',
            'مع', 'وقد', 'فقد', 'كان', 'كانت', 'كذلك', 'لكن', 'لذلك',
            'و', 'ف', 'ل', 'ب', 'ك'
        }

        # Add NLTK stopwords if available
        if NLTK_AVAILABLE:
            try:
                self.arabic_stopwords.update(set(stopwords.words('arabic')))
            except:
                print("Using basic Arabic stopwords")

    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text."""
        if PYARABIC_AVAILABLE:
            text = normalize_hamza(text)
            text = normalize_lamalef(text)
            text = normalize_tah_marbota(text)
        return text

    def remove_arabic_diacritics(self, text: str) -> str:
        """Remove Arabic diacritical marks."""
        if PYARABIC_AVAILABLE:
            text = araby.strip_tashkeel(text)
            text = araby.strip_tatweel(text)
        else:
            text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
        return text

    def remove_arabic_punctuation(self, text: str) -> str:
        """Remove Arabic punctuation."""
        arabic_punctuation = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ'''
        text = text.translate(str.maketrans('', '', arabic_punctuation))
        return text

    def preprocess(self, text: str) -> str:
        """Main preprocessing function."""
        if not text or not isinstance(text, str):
            return ""

        # Print debugging info
        print(f"Processing text: {text[:50]}...")

        try:
            # Normalize
            if self.normalize_arabic:
                text = self.normalize_arabic_text(text)

            # Remove diacritics
            if self.remove_diacritics:
                text = self.remove_arabic_diacritics(text)

            # Remove punctuation
            if self.remove_punctuation:
                text = self.remove_arabic_punctuation(text)

            # Remove numbers
            if self.remove_numbers:
                text = re.sub(r'\d+', ' ', text)

            # Split into words
            words = text.split()

            # Remove stopwords
            if self.remove_stopwords:
                words = [w for w in words if w not in self.arabic_stopwords]

            # Join words
            text = ' '.join(words)

            # Clean extra whitespace
            text = ' '.join(text.split())

            return text

        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return text

    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """Process a list of documents."""
        return [self.preprocess(doc) for doc in documents if doc]