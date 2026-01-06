import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .constants import EMBEDDING_PATH, SENTENCE_TRANSFORMER_MODEL
from .metrics_config import MetricsConfig


def embed_corpus(dataset,
                 embedder: str = None,
                 emb_filename: str = None,
                 emb_path: str = EMBEDDING_PATH,
                 save: bool = False,
):
    """
    Create a dictionary with the word embedding of every word in the dataset.
    Use the embedder. If the file 'Embeddings/{emb_filename}.pickle' is available, 
    read the embeddings from this file. Otherwise create new embeddings.
    Returns the embedding dict
    """
    # Check if embedder is a local path or model name and load accordingly
    if not embedder:
        embedder_name = MetricsConfig.SENTENCE_embedder or SENTENCE_TRANSFORMER_MODEL
        if os.path.exists(embedder_name):
            print(f"Loading model from local path: {embedder_name}")
            embedder = SentenceTransformer(embedder_name)
        else:
            print(f"Downloading model: {embedder_name}")
            embedder = SentenceTransformer(embedder_name)

    if emb_filename is None:
        emb_filename = str(dataset)
    try:
        emb_dic = pickle.load(open(f"{emb_path}{emb_filename}.pickle", "rb"))
    except FileNotFoundError:
        emb_dic = {}
        word_list = []
        for doc in dataset.get_corpus():
            for word in doc:
                word_list.append(word)

        word_list = set(word_list)
        for word in tqdm(word_list):
            emb_dic[word] = embedder.encode(word)

        if save:
            with open(f"{emb_path}{emb_filename}.pickle", "wb") as handle:
                pickle.dump(emb_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return emb_dic


def update_corpus_dic_list(
    word_lis: list,
    emb_dic: dict,
    embedder: str = None,
    emb_filename: str = None,
    emb_path: str = EMBEDDING_PATH,
    save: bool = False,
):
    """
    Updates embedding dict with embeddings in word_lis
    """

    # Check if embedder is a local path or model name and load accordingly
    if not embedder:
        embedder_name = MetricsConfig.SENTENCE_embedder or SENTENCE_TRANSFORMER_MODEL
        if os.path.exists(embedder_name):
            print(f"Loading model from local path: {embedder_name}")
            embedder = SentenceTransformer(embedder_name)
        else:
            print(f"Downloading model: {embedder_name}")
            embedder = SentenceTransformer(embedder_name)

    try:
        emb_dic = pickle.load(open(f"{emb_path}{emb_filename}.pickle", "rb"))
    except FileNotFoundError as e:
        print(e)
        print("No existing embedding found. Starting to embed corpus update dictionary")

        keys = set(emb_dic.keys())
        for word in tqdm(set(word_lis)):
            if word not in keys:
                emb_dic[word] = embedder.encode(word)

        if save:
            with open(f"{emb_path}{emb_filename}.pickle", "wb") as handle:
                pickle.dump(emb_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return emb_dic


def embed_topic(
    topics_tw,
    corpus_dict: dict,
    n_words: int = 10,
    embedder: str = None,
):
    """
    takes the list of topics and embed the top n_words words with the corpus dict
    if possible, else use the embedder.
    """
    # Check if embedder is a local path or model name and load accordingly
    if not embedder:
        embedder_name = MetricsConfig.SENTENCE_embedder or SENTENCE_TRANSFORMER_MODEL
        if os.path.exists(embedder_name):
            print(f"Loading model from local path: {embedder_name}")
            embedder = SentenceTransformer(embedder_name)
        else:
            print(f"Downloading model: {embedder_name}")
            embedder = SentenceTransformer(embedder_name)

    topic_embeddings = []
    for topic in tqdm(topics_tw):
        if n_words is not None:
            topic = topic[:n_words]

        add_lis = []
        for word in topic:
            try:
                add_lis.append(corpus_dict[word])
            except KeyError:
                # print(f'did not find key {word} to embedd topic, create new embedding...')
                add_lis.append(embedder.encode(word))

        topic_embeddings.append(add_lis)

    return topic_embeddings


def embed_stopwords(
    stopwords: list,
    embedder: str = None,
):
    """
    take the list of stopwords and embeds them with embedder
    """

    # Check if embedder is a local path or model name and load accordingly
    if not embedder:
        embedder_name = MetricsConfig.SENTENCE_embedder or SENTENCE_TRANSFORMER_MODEL
        if os.path.exists(embedder_name):
            print(f"Loading model from local path: {embedder_name}")
            embedder = SentenceTransformer(embedder_name)
        else:
            print(f"Downloading model: {embedder_name}")
            embedder = SentenceTransformer(embedder_name)

    sw_dic = {}  # first create dictionary with embedding of every unique word
    stopwords_set = set(stopwords)
    for word in stopwords_set:
        sw_dic[word] = embedder.encode(word)

    sw_list = []
    for word in stopwords:  # use this dictionary to embed all the possible stopwords
        sw_list.append(sw_dic[word])

    return sw_list


def mean_over_diag(mat):
    """
    Calculate the average of all elements of a quadratic matrix
    that are above the diagonal
    """
    h, w = mat.shape
    assert h == w, "matrix must be quadratic"
    mask = np.triu_indices(h, k=1)
    return np.mean(mat[mask])


def cos_sim_pw(mat):
    """
    calculate the average cosine similarity of all rows in the matrix (but exclude the similarity of a row to itself)
    """
    sim = cosine_similarity(mat)
    return mean_over_diag(sim)
