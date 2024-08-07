# STREAM
Multiple automated topic model evaluation metrics are available in STREAM.
Apart from classical NPMI, most of them are introduced in this [paper](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1). 


These are the availableMetrics
=================

| **Name**                                                                                                                                                 | **Description**                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ISIM](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)                | Average cosine similarity of top words of a topic to an intruder word.                                                                                                 |
| [INT](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)                 | For a given topic and a given intruder word, Intruder Accuracy is the fraction of top words to which the intruder has the least similar embedding among all top words. |
| [ISH](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)                 | calculates the shift in the centroid of a topic when an intruder word is replaced.                                                                                     |
| [Expressivity](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)        | Cosine Distance of topics to meaningless (stopword) embedding centroid                                                                                                 |
| [Embedding Topic Diversity](https://link.springer.com/chapter/10.1007/978-3-030-80599-9_4)                                                               | Topic diversity in the embedding space                                                                                                                                 |
| [Embedding Coherence](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1) | Cosine similarity between the centroid of the embeddings of the stopwords and the centroid of the topic.                                                               |
| [NPMI](https://aclanthology.org/E14-1056.pdf)                                                                                                            | Classical NPMi coherence computed on the scource corpus.                                                                                                               |


### Metric Evaluation

In this section, we describe the three metrics used to evaluate topic models' performance: **Intruder Shift (ISH)**, **Intruder Accuracy (INT)**, and **Average Intruder Similarity (ISIM)**.

### Expressivity
**Expressivity**,  evaluates the meaningfulness of a topic by leveraging stopwords. Stopwords primarily serve a grammatical role and don't contribute to the document's meaning. The steps to calculate Expressivity are as follows:

1. Compute vector embeddings for all stopwords and calculate their centroid embedding, $\bm{\psi}$.
2. For each topic, compute the weighted centroid of the top $Z$ words, normalized so that their weights sum up to 1: $\bm{\gamma}_k = \frac{1}{Z}\sum_{i=1}^{Z} \phi_{k,i}\bm{\omega_i}$.
3. Calculate the cosine similarity between each topic centroid $\bm{\gamma}_k$ and the stopword centroid $\bm{\psi}$.
4. The Expressivity metric is then defined as the average similarity across all $K$ topics:

$$\small{EXPRS(\bm{\gamma}, \bm{\psi}) = \frac{1}{K} \sum_{k=1}^{K} sim(\bm{\gamma}_k, \bm{\psi})}$$

Note that $\bm{\gamma}_k$ is different from $\bm{\mu}_k$, where the latter is the centroid of the document cluster associated with topic $t_k$. Expressivity can vary based on the chosen stopwords, allowing for domain-specific adjustments to evaluate a topic's expressivity based on a custom stopword set.

This approach provides a quantifiable measure of how well a topic conveys meaningful information, distinct from grammatical structure alone.


### Intruder Accuracy (INT)

The **Intruder Accuracy (INT)** metric aims to improve the identification of intruder words within a topic. Here's how it works:

1. Given the top Z words of a topic, randomly select an intruder word from another topic.
2. Calculate the cosine similarity between all possible pairs of words within the set of the top Z words and the intruder word.
3. Compute the fraction of top words for which the intruder has the least similar word embedding using the following formula:
 
$$\small{INT(t_k) = \frac{1}{Z}\sum_{i=1}^Z {1}(\forall j: sim(\bm{\omega}_i, \bm{\hat{\omega}}) < sim(\bm{\omega}_i, \bm{\omega}_j))}$$


INT measures how effectively the intruder word can be distinguished from the top words in a topic. A larger value is better.

### Average Intruder Similarity (ISIM)

The **Average Intruder Similarity (ISIM)** metric calculates the average cosine similarity between each word in a topic and an intruder word:
$$ISIM(t_k) = \frac{1}{Z} \sum_{i=1}^{Z} sim(\bm{\omega}_i, \bm{\hat{\omega}})$$

To enhance the metrics' robustness against the specific selection of intruder words, ISH, INT, and ISIM are computed multiple times with different randomly chosen intruder words, and the results are averaged.

These metrics provide insights into the performance of topic models and their ability to maintain topic coherence and diversity. A smaller value is better.

### Intruder Shift (ISH)

The **Intruder Shift (ISH)** metric quantifies the shift in a topic's centroid when an intruder word is substituted. This process involves the following steps:

1. Compute the unweighted centroid of a topic and denote it as $\tilde{\boldsymbol{\gamma}}_i$.
2. Randomly select a word from that topic and replace it with a randomly selected word from a different topic.
3. Recalculate the centroid of the resulting words and denote it as $\hat{\boldsymbol{\gamma}}_i$.
4. Calculate the ISH score for a topic by averaging the cosine similarity between $\tilde{\bm{\gamma}}_i$ and $\hat{\boldsymbol{\gamma}}_i$ for all topics using the formula:
5. 
$$ISH(T) = \frac{1}{K} \sum_{i=1}^{K} sim(\tilde{\bm{\gamma}}_i, \hat{\bm{\gamma}}_i)$$
A lower ISH score indicates a more coherent and diverse topic model.



The correlation of the intruder based metrics with human detection of intruder words are given here:

| Score                          | Intruder | **Human** | Intruder | **Human** |
| ------------------------------ | -------- | --------- | -------- | --------- |
| **Paraphrase-MiniLM-L6-v2**    |          |           |          |           |
| $ISH$                          | 0.613    | 0.512     | 0.526    | 0.492     |
| $INT$                          | 0.722    | **0.622** | 0.775    | **0.728** |
| $ISIM$                         | 0.810    | **0.686** | 0.574    | **0.539** |
| **Multi-qa-mpnet-base-dot-v1** |          |           |          |           |
| $ISH$                          | 0.675    | 0.573     | 0.598    | 0.567     |
| $INT$                          | 0.700    | **0.604** | 0.751    | 0.708     |
| $ISIM$                         | 0.791    | 0.672     | 0.543    | 0.511     |
| **All-MiniLM-L12-v2**          |          |           |          |           |
| $ISH$                          | 0.766    | **0.652** | 0.519    | **0.591** |
| $INT$                          | 0.677    | 0.580     | 0.723    | 0.687     |
| $ISIM$                         | 0.766    | **0.652** | 0.519    | 0.490     |
| **All-mpnet-base-v2**          |          |           |          |           |
| $ISH$                          | 0.763    | **0.652** | 0.626    | **0.592** |
| $INT$                          | 0.661    | 0.577     | 0.727    | 0.689     |
| $ISIM$                         | 0.763    | **0.652** | 0.511    | 0.482     |
| **All-distilroberta-v1**       |          |           |          |           |
| $ISH$                          | 0.766    | **0.652** | 0.625    | **0.592** |
| $INT$                          | 0.677    | 0.587     | 0.729    | 0.687     |
| $ISIM$                         | 0.766    | **0.652** | 0.519    | 0.490     |
| **word2vec GoogleNews**        |          |           |          |           |
| $ISH$                          | 0.413    | 0.335     | 0.338    | 0.302     |
| $INT$                          | 0.719    | 0.603     | 0.774    | **0.715** |
| $ISIM$                         | 0.820    | **0.684** | 0.554    | **0.506** |
| **Glove Wikipedia**            |          |           |          |           |
| $ISH$                          | 0.622    | 0.506     | 0.496    | 0.439     |
| $INT$                          | 0.750    | **0.634** | 0.786    | **0.727** |
| $ISIM$                         | 0.808    | **0.677** | 0.595    | **0.549** |


