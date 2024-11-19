# from stream_topic.models import KmeansTM
# from stream_topic.utils import TMDataset


# #本段落用时9min
# dataset = TMDataset()
# dataset.fetch_dataset(name = "my_dataset", dataset_path = "/hongyi/stream/dataset", source = 'local')
# dataset.preprocess(model_type="KmeansTM")
# #本段落用时4h
# model = KmeansTM('/hongyi/stream/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# model.fit(dataset, n_topics=10)


# topics = model.get_topics()
# print(topics)
# from stream_topic.metrics import ISIM
# from sentence_transformers import SentenceTransformer
# metric = ISIM()
# from sentence_transformers import SentenceTransformer

# metric = ISIM(metric_embedder_name='/hongyi/stream/sentence-transformers/Conan-embedding-v1/')
# metric.score(topics)
from stream_topic.metrics import ISIM, INT, ISH,Expressivity, NPMI, Embedding_Coherence, Embedding_Topic_Diversity
metric = Embedding_Coherence()


