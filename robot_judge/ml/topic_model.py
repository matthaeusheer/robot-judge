from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


def train_lda(list_of_words, n_topics):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(list_of_words)
    lda = LDA(n_components=n_topics)
    lda.fit(count_data)
    return lda, count_vectorizer, count_data


def print_topics(model, count_vectorizer, n_top_words):
    print(f'Printing {n_top_words} top words for all topics.')
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print(f'\n --- Topic {topic_idx} ---')
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))