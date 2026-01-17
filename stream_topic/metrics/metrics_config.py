# metrics_config.py
class MetricsConfig:
    PARAPHRASE_embedder = None  # Default to None, will be set dynamically
    SENTENCE_embedder = None    # Default to None, will be set dynamically
    language = "english"        # Default to "english", can be set to "chinese"

    @classmethod
    def set_PARAPHRASE_embedder(cls, name_path):
        cls.PARAPHRASE_embedder = name_path

    @classmethod    
    def set_SENTENCE_embedder(cls, name_path):
        cls.SENTENCE_embedder = name_path
        
    @classmethod    
    def set_language(cls, language):
        cls.language = language
