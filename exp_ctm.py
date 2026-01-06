from stream_topic.models import KmeansTM,BERTopicTM,CBC,DCTE,NMFTM,SOMTM,CEDC,ETM,LDA,ProdLDA,SOMTM,NSTM,WordCluTM,CTM,TNTM,NeuralLDA,CTMNeg
from stream_topic.utils import TMDataset
#本段落用时9min
dataset = TMDataset(language="chinese", stopwords_path = '/hongyi/stream/stopwords/common_stopwords.txt')# 
dataset.fetch_dataset(name = "Toutiao_imbalanced", dataset_path = "/hongyi/stream/dataset/paper_data", source = 'local')#
dataset.preprocess(model_type="CTM", min_word_length = 1)
from stream_topic.metrics import ISIM, INT, ISH,Expressivity, NPMI, Embedding_Coherence, Embedding_Topic_Diversity
from sentence_transformers import SentenceTransformer
from stream_topic.metrics.metrics_config import MetricsConfig
MetricsConfig.set_PARAPHRASE_embedder("/hongyi/stream/sentence-transformers/Conan-embedding-v1/")#paraphrase-multilingual-mpnet-base-v2
MetricsConfig.set_SENTENCE_embedder("/hongyi/stream/sentence-transformers/Conan-embedding-v1/")#all-mpnet-base-v2
best_params={'best_params': {
  'lr': 0.008298001572518747,
  'weight_decay': 0.0004044088999371794}}
import pandas as pd
import numpy as np
total_topics, NPMI_topics = [], []
ISIM1, INT1, ISH1, WESS1, EXPRS1, NPMI1, COH1 = [], [], [], [], [], [], []
for i in range(1):
    model = CTM(embedding_model_name="/hongyi/stream/sentence-transformers/Conan-embedding-v1/")
    # model = NMFTM(stopwords_path = '/hongyi/stream/stopwords/common_stopwords.txt',hparams=best_params)
    model.hparams.update(best_params['best_params'])
    n=14
    # model.fit(dataset,n_topics=n)#, language = "chinese"
    model.fit(dataset,n_topics=n, language = "chinese",**best_params)#
    
    topics = model.get_topics()
    total_topics.append(topics)
    
    score_list=[]
    metric = ISIM()
    for i in range(100):    
        scores = metric.score(topics) #值越小越好
        score_list.append(scores)
    ISIM1.append(np.mean(score_list))
    score_list=[]
    metric = INT()
    for i in range(100):    
        scores = metric.score(topics) #值越大越好
        score_list.append(scores)
    INT1.append(np.mean(score_list))
    score_list=[]
    metric = ISH()
    for i in range(100):    
        scores = metric.score(topics) #值越小越好
        score_list.append(scores)
    ISH1.append(np.mean(score_list))
    beta = np.random.rand(n, 384)
    diversity_metric = Embedding_Topic_Diversity()
    scores = diversity_metric.score(topics, beta)  #值越小越好
    WESS1.append(scores)
    expressivity_metric = Expressivity(
    n_words=10,
    custom_stopwords='/hongyi/stream/stopwords/common_stopwords.txt'
    )
    scores = expressivity_metric.score(topics, beta) #值越小越好
    EXPRS1.append(scores)
    metric = NPMI(dataset,language = "chinese", stopwords='/hongyi/stream/stopwords/common_stopwords.txt') #值越大越好    
    scores = metric.score(topics)  #值越大越好
    NPMI1.append(scores)
    metric = NPMI(dataset,language = "chinese", stopwords='/hongyi/stream/stopwords/common_stopwords.txt') #值越大越好    
    scores2 = metric.score_per_topic(topics)  #值越大越好
    NPMI_topics.append(scores2)
    metric = Embedding_Coherence()
    overall_score = metric.score(topics)
    COH1.append(overall_score)

metrics = {'ISIM':ISIM1, 'INT':INT1, 'ISH':ISH1, 'WESS':WESS1, 'EXPRS':EXPRS1, 'NPMI':NPMI1, 'COH':COH1}
df = pd.DataFrame(metrics).transpose()
df.to_csv('/hongyi/STREAM/result/benchmark/Toutiao/CTM_metrics_imb.csv')
df2 = pd.DataFrame(total_topics) 
df2.to_csv('/hongyi/STREAM/result/benchmark/Toutiao/CTM_topics_imb.csv')
df3 = pd.DataFrame(NPMI_topics) 
df3.to_csv('/hongyi/STREAM/result/benchmark/Toutiao/CTM_NPMI_imb.csv')