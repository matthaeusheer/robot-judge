import os
import itertools
from collections import Counter
import pandas as pd

from gensim.models.phrases import Phrases

from robot_judge.utils.data_structs import flatten, get_most_n_most_common_counter_entries
import robot_judge.nlp.filter as filters


def aggregate_clean_sentences(doc_dict):
    """From a corpus dict (label-raw_text key value pairs), construct a list of sentences (list of list of words).

    Returns
    -------
    sentences: A list of lists, where each inner list is a sentence split into words, i.e. a list of words.
    """

    sentences = {}
    for key, doc in doc_dict.items():
        sentences[key] = []
        for sentence in doc.sents:
            sentences[key].append([token.lower_ for token in sentence if
                                   not filters.token_is_punct_space(token)
                                   and not filters.token_is_stopword(token)])
    return sentences


def train_phrase_model(sentences, min_count=5):
    """Given a list of sentences, train a gensim phrase model."""

    phrase_model = Phrases(sentences, min_count=min_count)

    return phrase_model


def print_label_sent_dict(label_sent_dict):
    """For debugging. Takes a dictionary of (case) labels and sentences and prints out to console."""

    for label, sentences in label_sent_dict.items():
        print('Trigram sentences for case: ', label)
        for sent in sentences:
            print(sent)
            print()
        print(10*'-')


def create_df_from_label_sent_dict(label_sent_dict):
    """Takes a dictionary of label keys and sentence values and creates a BoW pandas data frame.

    For each (case) label, the count of all words is being accumulated and
    then a pandas data frame is being constructed.
    """

    counter_dict = {}
    for label, sentences in label_sent_dict.items():
        all_words = itertools.chain(*sentences)
        counter = Counter(all_words)
        counter_dict[label] = counter

    feat_df = pd.DataFrame(counter_dict).transpose()

    return feat_df


def get_sents_from_sentence_dict(sentence_dict):
    """List over all cases in sentence dict where each sublist holds lists of words (sentences)."""
    return [sent for sent in sentence_dict.values()]


def get_most_common_words(label_sent_dict, n_most_common=1000):
    """Having a label - sentences dictionary, get the n_most_common words seen over all sentences of all cases."""

    sentences = get_sents_from_sentence_dict(label_sent_dict)

    words = list(flatten(sentences))

    word_counter = Counter(words)

    return get_most_n_most_common_counter_entries(word_counter, n_most_common)


def get_labels_without_year(df):
    """The Labels of cases do have a _<year> - appendix in the name. This method truncates this appendix away."""
    return list([label.split('_')[1] for label in df.index])


def filter_words(trigram_sentence_dict, n_most_common=1000):
    """This function extracts the words which are being used as features in the classification.

    Step 1: Take all the bi- and tri-gram words (words which hold one or two underline characters).
    Step 2: Fill up the feature space which usual words which are the most common ones over the whole corpus.

    """
    all_words = list(flatten(trigram_sentence_dict.values()))
    bi_tri_words = list(set([word for word in all_words if word.count('_') in [1, 2]]))

    most_common_bi_tri_words = get_most_n_most_common_counter_entries(Counter(bi_tri_words), n_most_common)

    if len(most_common_bi_tri_words) == n_most_common:
        return most_common_bi_tri_words[:n_most_common]
    else:
        most_common_words_all = get_most_common_words(trigram_sentence_dict, n_most_common)

        for word in most_common_words_all:
            if word not in most_common_bi_tri_words:
                most_common_bi_tri_words.append(word)

    return most_common_bi_tri_words
