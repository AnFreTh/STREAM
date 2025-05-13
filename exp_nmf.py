from stream_topic.models import KmeansTM,BERTopicTM,CBC,DCTE,NMFTM,SOMTM,CEDC,ETM,LDA,ProdLDA,SOMTM,NSTM,WordCluTM,CTM,TNTM,NeuralLDA,CTMNeg
from stream_topic.utils import TMDataset
import numpy
print(numpy.__config__.show())  # 显示 BLAS 和 LAPACK 配置
#本段落用时9min
dataset = TMDataset()
# dataset.fetch_dataset(name="Stocktwits_GME", dataset_path = "/hongyi/STREAM/stream_topic/stream_topic_data/preprocessed_datasets/Stocktwits_GME",source = 'local')#
# dataset.preprocess(model_type="NMFTM")
# model = NMFTM()#
# # model = NMFTM(stopwords_path = '/hongyi/stream/stopwords/common_stopwords.txt')# 
# model.fit(dataset,n_topics=10)#

# topics = model.get_topics()
# print(topics)
dataset.fetch_dataset(name="UM_en",dataset_path = "/hongyi/stream/dataset",source = 'local')#
dataset.preprocess(model_type="KmeansTM")
model = KmeansTM(embedding_model_name="/hongyi/stream/sentence-transformers/paraphrase-MiniLM-L3-v2")#
model.fit(dataset,n_topics=7)
topics = model.get_topics()
print(topics)
# from stream_topic.metrics import ISIM, INT, ISH,Expressivity, NPMI, Embedding_Coherence, Embedding_Topic_Diversity
# from sentence_transformers import SentenceTransformer
# from stream_topic.metrics.metrics_config import MetricsConfig
# MetricsConfig.set_PARAPHRASE_embedder("/hongyi/stream/sentence-transformers/Conan-embedding-v1/")#paraphrase-multilingual-mpnet-base-v2
# MetricsConfig.set_SENTENCE_embedder("/hongyi/stream/sentence-transformers/Conan-embedding-v1/")#all-mpnet-base-v2
# import numpy as np
# import pandas as pd
# def load_stopwords(stopwords_path):
#     with open(stopwords_path, 'r', encoding='UTF-8') as f:
#         stopwords = [line.strip() for line in f]
#     return pd.DataFrame({'w': stopwords})
# stopword = load_stopwords('/hongyi/stream/stopwords/common_stopwords.txt')
# metric = NPMI(dataset,language = "chinese", custom_stopwords=list(stopword)) #值越大越好    
# scores = metric.score(topics)  #值越大越好
# print("NPMI score:", scores)
