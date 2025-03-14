from stream_topic.models import KmeansTM,BERTopicTM,CBC,DCTE,NMFTM,SOMTM,CEDC,ETM,LDA,ProdLDA,SOMTM,NSTM,WordCluTM,CTM,TNTM,NeuralLDA,CTMNeg
from stream_topic.utils import TMDataset

#本段落用时9min
dataset = TMDataset(language="chinese", stopwords_path = '/hongyi/stream/stopwords/common_stopwords.txt')# 
# dataset.fetch_dataset(name = "Fudan", dataset_path = "/hongyi/stream/dataset/paper_data", source = 'local')#THUCNews_small
dataset.fetch_dataset(name = "my_dataset", dataset_path = "/hongyi/stream/dataset", source = 'local')
dataset.preprocess(model_type="NMFTM", min_word_length = 1)
# model=KmeansTM(embedding_model_name="/hongyi/stream/sentence-transformers/Conan-embedding-v1/",stopwords_path = '/hongyi/stream/stopwords/common_stopwords.txt')
# model=LDA()
# model.fit(dataset,n_topics=14)
model = NMFTM(stopwords_path = '/hongyi/stream/stopwords/common_stopwords.txt')# 
model.fit(dataset,n_topics=10)#
# model = WordCluTM(word_embedding_model_name="/hongyi/stream/sentence-transformers/Conan-embedding-v1/")#
# model.fit(dataset,n_topics=2)

topics = model.get_topics()
print(topics)

from stream_topic.metrics import ISIM, INT, ISH,Expressivity, NPMI, Embedding_Coherence, Embedding_Topic_Diversity
from sentence_transformers import SentenceTransformer
from stream_topic.metrics.metrics_config import MetricsConfig
MetricsConfig.set_PARAPHRASE_embedder("/hongyi/stream/sentence-transformers/Conan-embedding-v1/")#paraphrase-multilingual-mpnet-base-v2
MetricsConfig.set_SENTENCE_embedder("/hongyi/stream/sentence-transformers/Conan-embedding-v1/")#all-mpnet-base-v2
import numpy as np
import pandas as pd
def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='UTF-8') as f:
        stopwords = [line.strip() for line in f]
    return pd.DataFrame({'w': stopwords})
stopword = load_stopwords('/hongyi/stream/stopwords/common_stopwords.txt')
metric = NPMI(dataset,language = "chinese", custom_stopwords=list(stopword)) #值越大越好    
scores = metric.score(topics)  #值越大越好
print("NPMI score:", scores)
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

model_output = {"topics": model.get_topics(), "topic-word-matrix": model.get_beta(), "topic-document-matrix": model.get_theta()}

metric = TopicDiversity(topk=10) # Initialize metric
topic_diversity_score = metric.score(model_output)
print(topic_diversity_score)
