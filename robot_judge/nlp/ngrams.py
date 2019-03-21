import os
import itertools
from collections import Counter
import pandas as pd

from robot_judge.nlp.language_models import spacy_nlp
from robot_judge.utils.data_structs import flatten_list_of_lists, get_most_n_most_common_counter_entries

import robot_judge.nlp.filter  as filters

from gensim.models.phrases import Phrases

from common import DATA_DIR_PATH


def aggregate_clean_sentences(corpus_dict):
    """From a corpus dict (label-raw_text key value pairs), construct a list of sentences.

    Returns
    -------
    sentences: A list of lists, where each inner list is a sentence split into words.
    """

    sentences = {}
    for key, text in corpus_dict.items():
        sentences[key] = []
        for sentence in spacy_nlp(text).sents:
            sentences[key].append([token.lower_ for token in sentence if
                                   not filters.token_is_punct_space(token)
                                   and not filters.token_is_stopword(token)])
    return sentences


def get_sents_from_sentence_dict(sentence_dict):
    return [sent for sent in sentence_dict.values()][0]


def train_phrase_model(sentences, min_count=5):

    phrase_model = Phrases(sentences, min_count=min_count)

    return phrase_model


def print_label_sent_dict(label_sent_dict):

    for label, sentences in label_sent_dict.items():
        print('Trigram sentences for case: ', label)
        for sent in sentences:
            print(sent)
            print()
        print(10*'-')


def create_df_from_label_sent_dict(label_sent_dict):

    counter_dict = {}
    for label, tri_sents in label_sent_dict.items():
        all_words = itertools.chain(*tri_sents)
        counter = Counter(all_words)
        counter_dict[label] = counter

    feat_df = pd.DataFrame(counter_dict).transpose()

    return feat_df


def get_most_common_words(label_sent_dict, n_most_common=1000):

    sentences = get_sents_from_sentence_dict(label_sent_dict)
    words = flatten_list_of_lists(sentences)

    word_counter = Counter(words)

    return get_most_n_most_common_counter_entries(word_counter, n_most_common)


def get_labels_without_year(df):
    return list([label.split('_')[1] for label in df.index])


def get_target_values(target_labels, data_dir='assignment_1', file_name='case_reversed.csv'):

    target_df = pd.read_csv(os.path.join(DATA_DIR_PATH, data_dir, file_name), index_col='caseid')
    all_case_targets = target_df.to_dict()['case_reversed']

    target_values = []
    for label in target_labels:
        target_values.append(all_case_targets[label])

    return target_values