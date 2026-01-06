<h1 style="text-align: center;">STREAM-ZH: Simplified Topic Retrieval, Exploration, and Analysis Module for Chinese language</h1>

<p>We extend STREAM and present STREAM-ZH, the first topic modeling package to fully support the Chinese language across a broad range of topic models, evaluation metrics, and preprocessing workflows.</p>


<h2> Table of Contents </h2>


- [🏃 Quick Start](#-quick-start)
- [🚀 Installation](#-installation)
- [📦 Available Models](#-available-models)
- [📊 Available Metrics](#-available-metrics)
- [🗂️ Available Datasets](#️-available-datasets)
- [🔧 Usage](#-usage)
  - [🛠️ Preprocessing](#️-preprocessing)
  - [🚀 Model fitting](#-model-fitting)
  - [✅ Evaluation](#-evaluation)
  - [🔍 Hyperparameter optimization](#-hyperparameter-optimization)
<!-- - [📜 Citation](#-citation) -->
- [📝 License](#-license)


# 🏃 Quick Start

Get started with STREAM-ZH in just a few lines of code:

```python
from stream_topic.models import KmeansTM
from stream_topic.utils import TMDataset

dataset = TMDataset(language="chinese", stopwords_path = 'stream_topic/utils/common_stopwords.txt')
dataset.fetch_dataset("THUCNews_small")
dataset.preprocess(model_type="KmeansTM")

model = KmeansTM(embedding_model_name="TencentBAC/Conan-embedding-v1", stopwords_path = 'stream_topic/utils/common_stopwords.txt')
model.fit(dataset, n_topics=14, language = "chinese")

topics = model.get_topics()
print(topics)
```


# 🚀 Installation

You can install STREAM-ZH directly from PyPI:
 ```bash
pip install stream_topic
```

# 📦 Available Models
STREAM-ZH inherits various neural and non-neural topic models provided by STREAM. Currently, the following models are implemented:

<div align="center" style="width: 100%;">
  <table style="margin: 0 auto;">
    <thead>
      <tr>
        <th><strong>Name</strong></th>
        <th><strong>Implementation</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?ref=http://githubhelp.com">LDA</a></td>
        <td>Latent Dirichlet Allocation</td>
      </tr>
      <tr>
        <td><a href="https://www.nature.com/articles/44565">NMF</a></td>
        <td>Non-negative Matrix Factorization</td>
      </tr>
      <tr>
        <td><a href="https://arxiv.org/abs/2004.14914">WordCluTM</a></td>
        <td>Tired of topic models?</td>
      </tr>
      <tr>
        <td><a href="https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1">CEDC</a></td>
        <td>Topics in the Haystack</td>
      </tr>
      <tr>
        <td><a href="https://arxiv.org/pdf/2212.09422.pdf">DCTE</a></td>
        <td>Human in the Loop</td>
      </tr>
      <tr>
        <td><a href="https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1">KMeansTM</a></td>
        <td>Simple Kmeans followed by c-tfidf</td>
      </tr>
      <tr>
        <td><a href="https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b3c81b523b1f03c87192aa2abbf9ffb81a143e54">SomTM</a></td>
        <td>Self organizing map followed by c-tfidf</td>
      </tr>
      <tr>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/10066754">CBC</a></td>
        <td>Coherence based document clustering</td>
      </tr>
      <tr>
        <td><a href="https://arxiv.org/pdf/2403.03737">TNTM</a></td>
        <td>Transformer-Representation Neural Topic Model</td>
      </tr>
      <tr>
        <td><a href="https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00325/96463/Topic-Modeling-in-Embedding-Spaces">ETM</a></td>
        <td>Topic modeling in embedding spaces</td>
      </tr>
      <tr>
        <td><a href="https://arxiv.org/abs/2004.03974">CTM</a></td>
        <td>Combined Topic Model</td>
      </tr>
      <tr>
        <td><a href="https://arxiv.org/abs/2303.14951">CTMNeg</a></td>
        <td>Contextualized Topic Models with Negative Sampling</td>
      </tr>
      <tr>
        <td><a href="https://arxiv.org/abs/1703.01488">ProdLDA</a></td>
        <td>Autoencoding Variational Inference For Topic Models</td>
      </tr>
      <tr>
        <td><a href="https://arxiv.org/abs/1703.01488">NeuralLDA</a></td>
        <td>Autoencoding Variational Inference For Topic Models</td>
      </tr>
      <tr>
        <td><a href="https://arxiv.org/abs/2008.13537">NSTM</a></td>
        <td>Neural Topic Model via Optimal Transport</td>
      </tr>
    </tbody>
  </table>
</div>



# 📊 Available Metrics
Since evaluating topic models, especially automatically, STREAM-ZH implements numerous evaluation metrics. Especially, the intruder based metrics, while they might take some time to compute, have shown great correlation with human evaluation. 
<div align="center" style="width: 100%;">
  <table style="margin: 0 auto;">
  <thead>
    <tr>
      <th><strong>Name</strong></th>
      <th><strong>Description</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1">ISIM</a></td>
      <td>Average cosine similarity of top words of a topic to an intruder word.</td>
    </tr>
    <tr>
      <td><a href="https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1">INT</a></td>
      <td>For a given topic and a given intruder word, Intruder Accuracy is the fraction of top words to which the intruder has the least similar embedding among all top words.</td>
    </tr>
    <tr>
      <td><a href="https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1">ISH</a></td>
      <td>Calculates the shift in the centroid of a topic when an intruder word is replaced.</td>
    </tr>
    <tr>
      <td><a href="https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1">Expressivity</a></td>
      <td>Cosine Distance of topics to meaningless (stopword) embedding centroid</td>
    </tr>
    <tr>
      <td><a href="https://link.springer.com/chapter/10.1007/978-3-030-80599-9_4">Embedding Topic Diversity</a></td>
      <td>Topic diversity in the embedding space</td>
    </tr>
    <tr>
      <td><a href="https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1">Embedding Coherence</a></td>
      <td>Cosine similarity between the centroid of the embeddings of the stopwords and the centroid of the topic.</td>
    </tr>
    <tr>
      <td><a href="https://aclanthology.org/E14-1056.pdf">NPMI</a></td>
      <td>Classical NPMi coherence computed on the source corpus.</td>
    </tr>
  </tbody>
</table>
</div>




# 🗂️ Available Datasets
STREAM-ZH provides the following Chinese datasets for benchmark testing:
<div align="center" style="width: 100%;">
  <table style="margin: 0 auto;">
  <thead>
    <tr>
      <th>Name</th>
      <th># Docs</th>
      <th># Words</th>
      <th># Avg Length</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>THUCNews</td>
      <td>804,656</td>
      <td>395,432</td>
      <td>230.5</td>
      <td>Preprocessed THUCNews dataset</td>
    </tr>
    <tr>
      <td>THUCNews_small</td>
      <td>13,994</td>
      <td>40,865</td>
      <td>198.1</td>
      <td>A subset of THUCNews with 1,000 documents per category</td>
    </tr>
    <tr>
      <td>FUDANCNews</td>
      <td>9,526</td>
      <td>22,985</td>
      <td>422.5</td>
      <td>Originally for text classification, merged from its training and test sets</td>
    </tr>
    <tr>
      <td>TOUTIAO</td>
      <td>337,902</td>
      <td>57,616</td>
      <td>10.2</td>
      <td>Preprocessed a headline dataset</td>
    </tr>
    <tr>
      <td>TOUTIAO_small</td>
      <td>19,399</td>
      <td>12,777</td>
      <td>8.1</td>
      <td>A subset of TOUTIAO with 1,400 documents per category</td>
    </tr>
    <tr>
      <td>CMtMedQA_ten</td>
      <td>48,413</td>
      <td>22,404</td>
      <td>166.1</td>
      <td>Preprocessed a Chinese multi-round medical conversation corpus, by selecting ten medical themes</td>
    </tr>
    <tr>
      <td>CMtMedQA_small</td>
      <td>9,909</td>
      <td>12,885</td>
      <td>164.6</td>
      <td>A subset of CMtMedQA_ten with 1,000 documents per category</td>
    </tr>
  </tbody>
</table>
</div>

# 🔧 Usage
To use one of the available models for Chinese topic modeling, follow the simple steps below:
1. Import the necessary modules:

    ```python
    from stream_topic.models import KmeansTM
    from stream_topic.utils import TMDataset
    ```
## 🛠️ Preprocessing
2. Get your dataset and preprocess for your model:
    ```python
    dataset = TMDataset(language="chinese", stopwords_path = 'stream_topic/utils/common_stopwords.txt')
    dataset.fetch_dataset("THUCNews_small")
    dataset.preprocess(model_type="KmeansTM")
    ```

The specified model_type is optional and further arguments can be specified. Default steps are predefined for all included models.


## 🚀 Model fitting

3. Choose the model you want to use and train it:
   
    ```python
    model = KmeansTM(embedding_model_name="TencentBAC/Conan-embedding-v1", stopwords_path = 'stream_topic/utils/common_stopwords.txt')
    model.fit(dataset, n_topics=10, language = "chinese")
    ```

Depending on the model, check the documentation for hyperparameter settings. To get the topics, simply run:

4. Get the topics:
    ```python
    topics = model.get_topics()
    ```

## ✅ Evaluation

Specify the embedding model of Chinese

```python
from stream_topic.metrics.metrics_config import MetricsConfig
MetricsConfig.set_PARAPHRASE_embedder("TencentBAC/Conan-embedding-v1")
MetricsConfig.set_SENTENCE_embedder("TencentBAC/Conan-embedding-v1")
```

To evaluate your model simply use one of the metrics.

```python
from stream_topic.metrics import ISIM, INT, ISH, Expressivity, NPMI

metric = ISIM()
metric.score(topics)
```

Scores for each topic are available via:
```python
metric.score_per_topic(topics)
```

## 🔍 Hyperparameter optimization
If you want to optimize the hyperparameters, simply run:
```python
model.optimize_and_fit(
    dataset,
    min_topics=2,
    max_topics=20,
    criterion="aic",
    n_trials=20,
)
```

# 📝 License

STREAM-ZH is released under the [MIT License](./LICENSE). © 2025 