import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ._embedder import BaseEmbedder


def clean_topics(topics, embedding_model, similarity=0.75):
    """Cleans the topics based on their cosine similarity between
    all words in the topic. Although we are only extracting nouns, and lemmatize them, it is possible that e.g.
    "tiger" and "tigers" are the top words in a topic. Therefore it could also happen, that
    all possible Conjugations of a word are the top k words from a topic.
    This would not be very meaningful/expressive. Hence we clean the topics.

    For each topic, iterates through every word an computes al cosine similarities between all words.
    It is cleaned top-down, which means, the first word will never be cleaned.
    If for instance the cosine similarity between word1 and word2 is larger than the specified threshold,
    word2 will be removed from the topic. If then the cosine similarity between word2 and word5 is also bigger than the
    specified threshold, word5 will remain in the corpus, as word2 is already removed.

    We compute all combinations between all words, hence a topic of k words, has (k-1)*((k-1)+1)/2 combinations.

    The resulting topics can hence vary in their lengths.


    Args:
        topics (_type_): the models topics
        embedding_model (_type_): BAse_Embedder class, see backend._base.py
        similarity (float, optional): cosine similarity threshold. Defaults to 0.75.

    Returns:
        dict: cleaned topics
    """

    word_embedding_model = BaseEmbedder(embedding_model)

    cleaned_topics = []
    # iterate through every topic
    for topic in tqdm(range(len(topics))):
        # extract words from topics
        keys = [word for t in topics[topic]
                for word in t if isinstance(word, str)]

        keys_ = word_embedding_model.create_word_embeddings(keys)

        mat = cosine_similarity(keys_)

        np.fill_diagonal(mat, 0)
        mat = np.triu(mat)

        sim_mat = np.where(mat >= similarity)

        word_list = []
        for a, b in zip(sim_mat[0], sim_mat[1]):
            word_list.append((keys[a], keys[b]))

        drop_values = []
        for i in range(len(word_list)):
            if word_list[i][0] not in drop_values:
                drop_values.append(word_list[i][1])

        k = [key for key in topics[topic] if key[0] not in drop_values]
        cleaned_topics.append(k)

    # create dictionary of cleaned topics
    dict_tops = {}
    for i in range(len(cleaned_topics)):
        dict_tops[i] = cleaned_topics[i]

    topic_mean_embeddings = []
    for k in range(len(dict_tops)):
        temp = 0
        words = [word for t in dict_tops[k]
                 for word in t if isinstance(word, str)]
        weights = [
            weight for t in dict_tops[k] for weight in t if isinstance(weight, float)
        ]
        weights = [weight / sum(weights) for weight in weights]
        for i in range(len(words)):
            temp += word_embedding_model.create_word_embeddings(
                words[i]) * weights[i]
        temp /= len(words)
        topic_mean_embeddings.append(temp)

    return dict_tops, topic_mean_embeddings
