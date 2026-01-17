<h1 style="text-align: center;">STREAM-ZH: Simplified Topic Retrieval, Exploration, and Analysis Module for Chinese language</h1>

<p>We extend STREAM and present STREAM-ZH, the first topic modeling package to fully support the Chinese language across a broad range of topic models, evaluation metrics, and preprocessing workflows.</p>


<h2> Table of Contents </h2>

- [🚀 Installation](#-installation)
- [📦 Available Models](#-available-models)
- [📊 Available Metrics](#-available-metrics)
- [🗂️ Available Datasets](#️-available-datasets)
- [🔧 Usage](#-usage)
  - [🛠️ Preprocessing](#️-preprocessing)
  - [🚀 Model fitting](#-model-fitting)
  - [✅ Evaluation](#-evaluation)

# 🚀 Installation

You can install STREAM-ZH directly from PyPI:
```python
pip install stream_topic
```
Please note that additional packages required for processing Chinese datasets may need to be installed
```python
pip install jieba
pip install hanlp
pip install thulac
pip install snownlp
pip install pkuseg
pip install opencc
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
        <td>LDA</td>
        <td>Latent Dirichlet Allocation</td>
      </tr>
      <tr>
        <td>NMF</td>
        <td>Non-negative Matrix Factorization</td>
      </tr>
      <tr>
        <td>WordCluTM</td>
        <td>Tired of topic models?</td>
      </tr>
      <tr>
        <td>CEDC</td>
        <td>Topics in the Haystack</td>
      </tr>
      <tr>
        <td>DCTE</td>
        <td>Human in the Loop</td>
      </tr>
      <tr>
        <td>KMeansTM</td>
        <td>Simple Kmeans followed by c-tfidf</td>
      </tr>
      <tr>
        <td>SomTM</td>
        <td>Self organizing map followed by c-tfidf</td>
      </tr>
      <tr>
        <td>CBC</td>
        <td>Coherence based document clustering</td>
      </tr>
      <tr>
        <td>TNTM</td>
        <td>Transformer-Representation Neural Topic Model</td>
      </tr>
      <tr>
        <td>ETM</td>
        <td>Topic modeling in embedding spaces</td>
      </tr>
      <tr>
        <td>CTM</td>
        <td>Combined Topic Model</td>
      </tr>
      <tr>
        <td>CTMNeg</td>
        <td>Contextualized Topic Models with Negative Sampling</td>
      </tr>
      <tr>
        <td>ProdLDA</td>
        <td>Autoencoding Variational Inference For Topic Models</td>
      </tr>
      <tr>
        <td>NeuralLDA</td>
        <td>Autoencoding Variational Inference For Topic Models</td>
      </tr>
      <tr>
        <td>NSTM</td>
        <td>Neural Topic Model via Optimal Transport</td>
      </tr>
    </tbody>
  </table>
</div>

# 📊 Available Metrics
STREAM-ZH inherits all the evaluation metrics of STREAM, including intruder, diversity and coherence metrics.
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
      <td>ISIM</td>
      <td>Average cosine similarity of top words of a topic to an intruder word.</td>
    </tr>
    <tr>
      <td>INT</td>
      <td>For a given topic and a given intruder word, Intruder Accuracy is the fraction of top words to which the intruder has the least similar embedding among all top words.</td>
    </tr>
    <tr>
      <td>ISH</td>
      <td>Calculates the shift in the centroid of a topic when an intruder word is replaced.</td>
    </tr>
    <tr>
      <td>Expressivity</td>
      <td>Cosine Distance of topics to meaningless (stopword) embedding centroid</td>
    </tr>
    <tr>
      <td>Embedding Topic Diversity</td>
      <td>Topic diversity in the embedding space</td>
    </tr>
    <tr>
      <td>Embedding Coherence</td>
      <td>Cosine similarity between the centroid of the embeddings of the stopwords and the centroid of the topic.</td>
    </tr>
    <tr>
      <td>NPMI</td>
      <td>Classical NPMi coherence computed on the source corpus.</td>
    </tr>
  </tbody>
</table>
</div>




# 🗂️ Available Datasets
STREAM-ZH provides the following preprocessed Chinese datasets for benchmark testing:
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
      <td>Preprocessed a news headline dataset</td>
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
2. Get the dataset and preprocess for your model:
    ```python
    dataset = TMDataset(language="chinese", stopwords_path = 'stream_topic/utils/common_stopwords.txt')
    dataset.fetch_dataset("THUCNews_small", dataset_path = "stream_ZH_topic_data/preprocessed_datasets/THUCNews", source = 'local')
    dataset.preprocess(model_type="KmeansTM")
    ```

The specified model_type is optional and further arguments can be specified. Default steps are predefined for all included models.


## 🚀 Model fitting

3. Choose the model you want to use and train it:
   
    ```python
    model = KmeansTM(embedding_model_name="TencentBAC/Conan-embedding-v1", stopwords_path = 'stream_topic/utils/common_stopwords.txt')# 
    model.fit(dataset, n_topics=14, language = "chinese")
    ```

To get the topics, simply run:

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

```python
metric = NPMI(dataset, language = "chinese", stopwords = 'stream_topic/utils/common_stopwords.txt')
metric.score(topics)
```

Scores for each topic are available via:
```python
metric.score_per_topic(topics)
```
