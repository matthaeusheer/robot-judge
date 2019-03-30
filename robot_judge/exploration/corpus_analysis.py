import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import dill as pickle

from robot_judge.nlp.language_models import spacy_nlp_en
from robot_judge.nlp.spacy import count_sents
from robot_judge.nlp.spacy import count_tokens
import robot_judge.nlp.filter as filters
from robot_judge.utils.numerics import round_up
from robot_judge.utils.data_structs import normalize_counter
from robot_judge.utils.timer import timeit
from robot_judge import io

from typing import Callable

from common import DATA_DIR_PATH


@timeit
def process_corpus_dict_to_doc_dict(corpus_dict, n_threads=4, batch_size=10, store=False, data_sub_dir=None):
    """Takes in a corpus dict (labelled texts) and runs spaCy on it.

    Returns
    -------
    doc_dict: Dictionary with same structure as corpus_dict but the values are now spacy docs instead of texts.
    """
    labels = list(corpus_dict.keys())
    n_batches = len(labels) / batch_size

    doc_dict = {}

    idx = 0
    for doc in spacy_nlp_en.pipe(corpus_dict.values(), n_threads=n_threads, batch_size=batch_size):
        doc_dict[labels[idx]] = doc
        idx += 1

        if idx % batch_size == 0:
            batch_idx = idx / batch_size
            print('Finished {} of {} batches.'.format(batch_idx, n_batches))

            if store:
                assert data_sub_dir is not None, 'Need to specify data sub directory if store = True.'
                io.create_folder_if_not_exists(os.path.join(DATA_DIR_PATH, data_sub_dir))

                store_doc_dict(doc_dict, data_dir=os.path.join(DATA_DIR_PATH, data_sub_dir),
                               file_name='doc_dict_batch_{}.pkl'.format(batch_idx))
                doc_dict = {}

    return doc_dict


def store_doc_dict(doc_dict, data_dir=DATA_DIR_PATH, file_name='doc_dict.pkl'):
    """Pickles a doc dict to file."""
    with open(os.path.join(data_dir, file_name), 'wb') as out_file:
        pickle.dump(doc_dict, out_file)
        print('Stored doc dict to {}'.format(os.path.join(data_dir, file_name)))


@timeit
def load_doc_dict(data_dir=DATA_DIR_PATH, file_name='doc_dict.pkl'):
    """Loads a pickled doc dict from file."""
    try:
        with open(os.path.join(data_dir, file_name), 'rb')as in_file:
            print('Loading doc dict {}'.format(os.path.join(data_dir, file_name)))
            return pickle.load(in_file)
    except OSError as e:
        print(e)
        print('There is no file {}. You have to process first and create this file.'.format(os.path.join(data_dir,
                                                                                                         file_name)))


def count_words_sents_letters(doc_dict):
    """Calculates the number of words, sents and letters for each entry in corpus_dict.

    Arguments
    ---------
    doc_dict: Dictionary of label - spacy_doc key-value pairs.

    Returns
    -------
    labels: List of keys of texts in the corpus_dict.
    word_counts: List of word counts for the associated keys in the titles list.
    sents_counts: List of sentence counts for the associated keys in the titles list.
    letters_counts: List of sentence counts for the associated keys in the titles list.

    """
    labels = []
    word_counts = []
    sents_counts = []
    letters_counts = []

    for label, doc in doc_dict.items():

        labels.append(label)

        # For the letters/sentence counting I basically take the raw unprocessed text and use spacy sentence detection.
        sents_counts.append(count_sents(doc))
        letters_counts.append(len(doc.text))

        # For word and letter count I apply some basic pre-processing like removing white spaces and punctuations.
        tok_filtered = [tok for tok in doc if not filters.token_is_punct_space(tok)]
        word_counts.append(count_tokens(tok_filtered))

    return labels, word_counts, sents_counts, letters_counts


def visualize_counts(years, word_counts, sents_counts, letters_counts):
    """Plots word_counts, sents_counts and letters_counts vs years."""

    df_dict = {'year': years, 'sents_count': sents_counts, 'words_count': word_counts, 'letters_count': letters_counts}
    counts_df = pd.DataFrame(df_dict)
    counts_df.sort_values('year', inplace=True)
    counts_df = counts_df.groupby(['year']).mean()  # Average counts per year

    axes = counts_df.plot(subplots=True, figsize=(13, 8), sharex=True, linewidth=2, legend=False)
    axes[0].set_ylabel('Mean number of sentences')
    axes[1].set_ylabel('Mean number of words')
    axes[2].set_ylabel('Mean number of letters')

    for i, ax in enumerate(axes):
        if i in [0, 1]:
            ax.xaxis.set_ticks_position('none')
        else:
            ax.xaxis.set_ticks((range(round_up(min(years), 10), round_up(max(years), 10), 10)))
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.show()


def get_pos_tags(doc_dict, transform_label: Callable=None):
    """For all entries in the corpus_dict, extract pos tags for all tokens in the texts.

    Arguments
    ---------
    doc_dict: Dictionary of label - spacy_doc key-value pairs.
    transform_label: Callable function to transform the labels from the corpus dict (e.g. extract year from label).

    Returns
    -------
    labels: Transformed or untransformed labels of the corpus_dict
    pos_tags: List of POS tags lists.
    """

    labels = []
    pos_tags = []

    for label, doc in doc_dict.items():
        if transform_label:
            label = transform_label(label)
        labels.append(label)

        pos_tags_ = [token.pos_ for token in doc]
        pos_tags.append(pos_tags_)

    return labels, pos_tags


def aggregate_avg_pos_tags(years, pos_tags):
    """Use all pos tags lists (which have to correspond to the years list),
    to calculate the average counts of pos tags per year.
    """

    sorted_set_years = sorted(set(years))
    avg_year_counters = []

    for year in sorted_set_years:

        pos_counter = Counter()
        n_samples_this_year = 0

        for idx, year_orig in enumerate(years):
            if year_orig == year:
                pos_counter += Counter(pos_tags[idx])
                n_samples_this_year += 1

        # Normalizing the counter
        pos_counter = normalize_counter(pos_counter)

        avg_year_counters.append(pos_counter)

    return pd.DataFrame(avg_year_counters, index=sorted_set_years)


def visualize_avg_pos_vs_year(pos_df):
    """Plot avg pos tag counter vs years."""
    _ = plt.figure()

    ax = pos_df.plot(figsize=(15, 8), legend=False)

    for line, name in zip(ax.lines, pos_df.columns):
        y = line.get_ydata()[-1]
        ax.annotate(name, xy=(1, y), xytext=(6, 0), color=line.get_color(),
                    xycoords=ax.get_yaxis_transform(), textcoords="offset points",
                    size=14, va="center")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_ylabel('Average counts frequency for POS tag')
    ax.set_xlabel('Year')

    plt.show()
