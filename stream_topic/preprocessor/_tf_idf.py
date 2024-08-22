import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """class based tf_idf retrieval from cluster of documents

    Args:
        documents (_type_): _description_
        m (_type_): _description_
        ngram_range (tuple, optional): _description_. Defaults to (1, 1).

    Returns:
        _type_: _description_
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(
        documents
    )
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)

    # Suppress divide by zero warning
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.divide(t.T, w)
        if np.any(np.isnan(tf)) or np.any(np.isinf(tf)):
            logger.warning("NaNs or inf in tf matrix")
            tf[~np.isfinite(tf)] = 0

    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_tfidf_topics(tf_idf, count, docs_per_topic, n=100):
    """class based tf_idf retrieval from cluster of documents

    Args:
        tf_idf (_type_): _description_
        count (_type_): _description_
        docs_per_topic (_type_): _description_
        n (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.predictions)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {
        label: [((words[j]), (tf_idf_transposed[i][j])) for j in indices[i]][::-1]
        for i, label in enumerate(labels)
    }

    return top_n_words


def extract_topic_sizes(df):
    """
    Extracts and computes the size of each topic from a given DataFrame.

    This function groups the DataFrame by the 'Topic' column, which represents
    topic IDs, and then counts the number of documents associated with each topic.
    It returns a DataFrame with two columns: 'Topic' and 'Size', where 'Size' is
    the count of documents in each topic. The returned DataFrame is sorted in
    descending order of 'Size'.

    Parameters:
        df (pandas.DataFrame): A DataFrame containing at least two columns, 'Topic'
                               and 'docs', where 'Topic' is an ID column for topics
                               and 'docs' contains documents or data points associated
                               with each topic.

    Returns:
        pandas.DataFrame: A DataFrame with 'Topic' and 'Size' columns, where 'Size'
                          indicates the number of documents in each topic, sorted in
                          descending order of size.
    """
    topic_sizes = (
        df.groupby(["Topic"])
        .docs.count()
        .reset_index()
        .rename({"Topic": "Topic", "docs": "Size"}, axis="columns")
        .sort_values("Size", ascending=False)
    )
    return topic_sizes
