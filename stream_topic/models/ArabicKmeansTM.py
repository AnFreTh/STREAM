"""
Arabic-Specific KMeans Topic Modeling Implementation

This implementation was created to address specific challenges in Arabic text processing:

1. Arabic Text Characteristics:
   - Complex morphology and word formation
   - Right-to-left script
   - Multiple forms of the same letter
   - Diacritical marks
   - Different word variations and normalizations

2. Stopwords Handling:
   - Enhanced Arabic stopwords list
   - Includes various forms of common Arabic words
   - Handles different Arabic character variations

3. Topic Extraction Improvements:
   - Better filtering for Arabic-specific patterns
   - Enhanced TF-IDF configuration for Arabic text
   - Improved word selection criteria

4. Processing Flexibility:
   - Supports both embedding-based and TF-IDF approaches
   - Specialized Arabic text preprocessing
   - Better handling of Arabic-specific tokenization

This implementation extends the base KmeansTM class with Arabic-specific optimizations
and should be used specifically for Arabic text corpus analysis.
"""

import nltk
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize
from loguru import logger
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from .KmeansTM import KmeansTM
from ..commons.check_steps import check_dataset_steps
from ..preprocessor import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import TrainingStatus

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "ArabicKmeansTM"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"

class ArabicKmeansTM(KmeansTM):
    """
    Arabic-specific implementation of KMeans topic modeling.
    Enhanced with better stopwords handling and topic extraction.
    """

    def __init__(
        self, 
        num_topics=10, 
        embedding_model_name: str = None,  # Optional embeddings for flexibility
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_topics = num_topics
        self.n_topics = num_topics
        self._status = TrainingStatus.NOT_STARTED
        
        # Store embedding model name if provided
        self.embedding_model_name = embedding_model_name
        
        # Core attributes
        self.clustering_model = None
        self.labels = None
        self.topic_dict = {} #  Initialize as empty dict instead of None
        self.dataframe = None
        self.beta = None
        self.theta = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.dataset = None  # Added for visualization support
        
        
        # Enhanced Arabic stopwords initialization
        self.arabic_stop_words = self._get_enhanced_stopwords()
        
    # NEW: Enhanced stopwords method
    def _get_enhanced_stopwords(self):
        """
        Get comprehensive Arabic stopwords list.
        
        Includes:
        - Basic NLTK Arabic stopwords
        - Additional common Arabic words
        - Various forms of verbs and their conjugations
        - Prepositions and conjunctions
        - Character variations (أ, إ, آ)
        """
        nltk.download('stopwords', quiet=True)
        basic_stops = set(stopwords.words('arabic'))
        
        # NEW: Additional Arabic stopwords
        additional_stops = {
            # Common verbs and their variations
            'كان', 'كانت', 'كانوا', 'يكون', 'تكون', 'اصبح', 'اصبحت',
            'قال', 'قالت', 'قالوا', 'يقول', 'تقول', 'وقال', 'واضاف',
            'اكد', 'اكدت', 'اوضح', 'اوضحت', 'اعلن', 'اعلنت',
            
            # Prepositions and conjunctions
            'نحو', 'حول', 'دون', 'حين', 'بين', 'بينما', 'منذ', 'خلال',
            'في', 'من', 'على', 'عن', 'مع', 'لدى', 'الى', 'بعد', 'قبل',
            
            # Pronouns and demonstratives
            'هو', 'هي', 'هم', 'هما', 'هن',
            'الذي', 'التي', 'الذين', 'اللتين', 'اللذين',
            
            # Other common stopwords
            'ان', 'إن', 'كما', 'لكن', 'لذلك', 'ايضا',
            'و', 'أو', 'ثم', 'أم', 'أن', 'لقد', 'إذ', 'إذا',
            'كل', 'وقد', 'فقد', 'نفى', 'ذكر', 'ذكرت',
            
            # Additional common words
            'يمكن', 'سوف', 'حيث', 'جدا', 'فقط', 'عندما', 'هناك',
            'الان', 'اليوم', 'امس', 'غدا', 'دائما', 'ابدا',
            'نعم', 'لا', 'ربما', 'هكذا', 'هنا', 'هنالك',
            'انه', 'انها', 'انهم', 'لماذا', 'كيف', 'متى', 'اين',
            'عند', 'فوق', 'تحت', 'امام', 'خلف', 'بجانب',
            'منها', 'منه', 'منهم', 'عنها', 'عنه', 'عنهم',
            'لها', 'له', 'لهم', 'بها', 'به', 'بهم'
        }
        # NEW: Add variations with different Arabic characters
        variations = set()
        for word in additional_stops:
            variations.add(word.replace('ا', 'أ'))
            variations.add(word.replace('ا', 'إ'))
            variations.add(word.replace('ا', 'آ'))
            
        return basic_stops.union(additional_stops).union(variations)

    # MODIFIED: Improved topic extraction with better filtering
    def _extract_topics(self, texts, labels):
        """
        Extract topics with enhanced filtering for Arabic text.
        
        Features:
        - Improved TF-IDF configuration for Arabic
        - Better word filtering criteria
        - Handles Arabic-specific patterns
        - Enhanced score thresholds
        """
        try:
            # Group documents by topic
            docs_by_topic = {}
            for text, label in zip(texts, labels):
                docs_by_topic.setdefault(label, []).append(text)
            
            # Join documents for each topic
            docs_per_topic = []
            for topic_idx in range(self.num_topics):
                topic_docs = docs_by_topic.get(topic_idx, [""])
                docs_per_topic.append(" ".join(topic_docs))
            
            # NEW: Enhanced TF-IDF configuration
            vectorizer = TfidfVectorizer(
                max_features=10000,
                lowercase=False,
                strip_accents=None,
                token_pattern=r'[^\s]+',
                stop_words=self.arabic_stop_words,
                min_df=2,  # NEW: Minimum document frequency
                max_df=0.95  # NEW: Maximum document frequency
            )
            
            # Calculate TF-IDF
            tfidf_matrix = vectorizer.fit_transform(docs_per_topic)
            feature_names = vectorizer.get_feature_names_out()

            # Store vectorizer vocabulary for beta matrix
            self.vocabulary = vectorizer.vocabulary_
            
            # Enhanced topic word extraction with filtering
            topics = []
            for topic_idx in range(self.num_topics):
                scores = tfidf_matrix[topic_idx].toarray()[0]
                top_indices = np.argsort(scores)[-50:][::-1]  # Get more candidates
                
                filtered_words = []
                for idx in top_indices:
                    word = feature_names[idx]
                    score = scores[idx]
                    
                    # NEW: Multiple filtering criteria
                    if (len(word) > 2 and  # Skip very short words
                        not any(char.isdigit() for char in word) and  # Skip numbers
                        word not in self.arabic_stop_words and  # Double-check stopwords
                        score > 0.01):  # Minimum score threshold
                        filtered_words.append((word, float(score)))
                        if len(filtered_words) >= 20:  # Keep top 20 after filtering
                            break
                
                topics.append(filtered_words)

                # Store in topic_dict
                self.topic_dict[topic_idx] = filtered_words

            return topics

        except Exception as e:
            logger.error(f"Error extracting Arabic topics: {e}")
            raise

    # MODIFIED: Updated fit method with flexible processing
    def fit(self, dataset: TMDataset):
        """
        Fit the model with enhanced Arabic processing.
        
        Supports:
        - Both embedding and TF-IDF approaches
        - Arabic-specific text preprocessing
        - Specialized clustering configuration
        """
        assert isinstance(dataset, TMDataset), "Dataset must be TMDataset instance"
        check_dataset_steps(dataset, logger, MODEL_NAME)

        self._status = TrainingStatus.INITIALIZED
        try:
            logger.info(f"Training {MODEL_NAME}")
            self._status = TrainingStatus.RUNNING
            
            # Store dataset for visualization
            self.dataset = dataset
            
            # Flexible processing based on embedding configuration
            if self.embedding_model_name:
                logger.info("Using embedding-based approach")
                self.dataset, self.embeddings = self._prepare_embeddings(dataset, logger)
                self.reduced_embeddings = self.dim_reduction(logger)
                X = self.reduced_embeddings
            else:
                logger.info("Using TF-IDF approach")
                vectorizer = TfidfVectorizer(
                    max_features=10000,
                    lowercase=False,
                    strip_accents=None,
                    token_pattern=r'[^\s]+',
                    stop_words=self.arabic_stop_words,
                    min_df=2,
                    max_df=0.95,
                )
                X = vectorizer.fit_transform(dataset.dataframe['text']).toarray()

            self.dataframe = dataset.dataframe.copy()
            
            # Perform clustering
            self.clustering_model = KMeans(
                n_clusters=self.num_topics,
                random_state=42,
                **self.kmeans_args
            )
            
            self.labels = self.clustering_model.fit_predict(X)
            
            # Extract topics
            self.topics = self._extract_topics(
                dataset.dataframe['text'],
                self.labels
            )
            
            # Create distributions
            self.dataframe['predictions'] = self.labels
            one_hot_encoder = OneHotEncoder(sparse=False)
            self.theta = one_hot_encoder.fit_transform(
                np.array(self.labels).reshape(-1, 1)
            )
            self.theta = normalize(self.theta, norm='l1', axis=1)
            
            if not hasattr(self, 'embeddings') or self.embeddings is None:
                self.embeddings = self.encode_documents(
                    dataset.dataframe['text'].tolist(),
                    encoder_model="paraphrase-MiniLM-L3-v2",
                    use_average=True
                )
                self.reduced_embeddings = self.dim_reduction(logger)
    
            logger.info("Training completed successfully")
            self._status = TrainingStatus.SUCCEEDED
            return self

        except Exception as e:
            logger.error(f"Training error: {e}")
            self._status = TrainingStatus.FAILED
            raise

    def get_topics(self, n_words=10):
        """Get the top n words for each topic."""
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model not trained")
            
        return [[word for word, _ in topic[:n_words]] 
                for topic in self.topics]
    
    def predict(self, texts):
        """Predict topics for new texts."""
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model not trained")

        # Process based on model type
        if self.embedding_model_name:
            embeddings = self.encode_documents(
                texts,
                encoder_model=self.embedding_model_name
            )
            X = self.reducer.transform(embeddings)
        else:
            vectorizer = TfidfVectorizer(
                max_features=10000,
                lowercase=False,
                strip_accents=None,
                token_pattern=r'[^\s]+',
                stop_words=self.arabic_stop_words
            )
            X = vectorizer.fit_transform(texts).toarray()
            
        return self.clustering_model.predict(X)

    # NEW: Helper method for visualization
    def visualize(self, port=8050):
        """Visualize the topic model."""
        from ..visuals import visualize_topic_model
        visualize_topic_model(
            self,
            reduce_first=False,
            port=port
        )