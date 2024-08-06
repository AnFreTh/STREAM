# STREAM
Multiple topic model/document clustering and subsequent topic extraction models are available in STREAM.
All models are kept conceptually simple and offer easy and fast computation. All are heavily drawing from [Sentence Transformers](https://arxiv.org/pdf/1908.10084.pdf). So if you use any of our models in a scientific poublication be sure to reference the used transformer model as well as the used model from the given links below:

## Available Models
================


### WordCluTM
----
[WordCluTM](https://arxiv.org/abs/2004.14914)    

Follows a simple word clustering approach based on Word2Vec.
We use the GMM approach introduced [Sia et al.](https://arxiv.org/abs/2004.14914) and get the doucment-topic-matrices by averaging over the topic words present in each document.

### CEDC
----

[CEDC](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)


Corpus Expansion and Document Clustering is a straightforward model that leverages word/sentence embeddings and Gaussian mixture models. The documents are embedded, their dimensionality is reduced (using [UMAP](https://arxiv.org/pdf/1802.03426.pdf?source=post_page---------------------------)), and then they are clustered using GMM. This process creates soft clusters and allows for the identification of documents that align with multiple topics or clusters.

After clustering, the candidate words for topics are extracted from the corpus (these could, for example, be only nouns). Additionally, these candidate words are enriched with words from additional corpora. This enrichment enables the creation of topics with words that are not present in the source corpus but better explain a given topic than the words included in the documents. The candidate words are then embedded, and topics are created by computing the cosine similarities between these candidate words and the cluster centroids.

### DCTE
----

[DCTE](https://arxiv.org/pdf/2212.09422.pdf)

Document classification and Topic Extraction is a semi-supervised model that can be fully unsupervised when leveraging e.g. automated labeling methods. It only takes very few labeled documents and might be a sensible option when the user already has an idea about certain topics in their corpus or deals with very obscure topics. 

The few labeled documents are used to train a few-shot classifier, namely [SetFit](https://arxiv.org/pdf/2209.11055.pdf). Subsequently the unlabeled documents are labelled by the classifier. From these labled documents, the topics are extracted using a class based tf-idf approach, similarly used in [BERTopic](https://arxiv.org/pdf/2203.05794.pdf).


### KMeansTM
--------

[KMeansTM](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1).

The most simple model of all available models, but often with very consistent and reliably good results. The documents are embedded, the dimensionality is reduced and they are clustered using KMeans. SUbsequently the topics are extracted from the clusters again with a class based tf-idf approach. 

### SomTM
-----

SomTM

A simple models based on Self organizing maps ([SOM in NLP](https://www.researchgate.net/profile/Timo-Honkela/publication/2749269_Self-Organizing_Maps_In_Natural_Language_Processing/links/09e4150fe6eed0f53b000000/Self-Organizing-Maps-In-Natural-Language-Processing.pdf)). Instead of dimensionality reduction using UMAP and subsequent clustering, the embeddings are clustered using the self-organizing map. The topics are again created using the class based tf-idf approach.

### CBC
---

[CBC](https://ieeexplore.ieee.org/abstract/document/10066754) 

An approach that completely works without document embeddings. This model leverages widely used coherence scores by integrating them into a novel document-level clustering approach that uses keyword extraction methods for small to medium sized datasets. The metric by which most topic extraction methods optimize their hyperparameters is thus optimized during clustering, resulting in coherent clusters. Moreover, unlike traditional methods, the number of extracted topics or clusters does not need to be determined in advance, saving an additional optimization step and a time- and computationally-intensive grid search. This implementation uses the louvain algorithm to create the clusters.





