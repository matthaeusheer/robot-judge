from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from robot_judge.nlp import count_sents
from robot_judge.nlp import count_tokens
from robot_judge.nlp import spacy_doc
import robot_judge.nlp.filter as filters
from robot_judge.utils.numerics import round_up
from robot_judge.utils.data_structs import normalize_counter

from typing import Callable


def count_words_sents_letters(corpus_dict):
    """Calculates the number of words, sents and letters for each entry in corpus_dict.

    Arguments
    ---------
    corpus_dict: Dictionary of label-(raw)text key-value pairs.

    Returns
    -------
    labels: List of keys of texts in the corpus_dict.
    word_counts: List of word counts for the associated keys in the titles list.
    sents_counts: List of sentence counts for the associated keys in the titles list.
    letters_counts: List of sentence counts for the associated keys in the titles list.

    """
    word_counts = []
    sents_counts = []
    letters_counts = []

    labels = []

    for label, case_text in corpus_dict.items():

        labels.append(label)
        # For the sentence counting I basically take the raw unprocessed text and use spacy sentence detection.
        sents_counts.append(count_sents(spacy_doc(case_text)))

        # For word and letter count I apply some basic preprocessing like removing white spaces and punctuations.
        doc = filters.remove_punct_and_sym(spacy_doc(case_text))
        doc = filters.remove_ws_tokens(doc)

        word_counts.append(count_tokens(doc))
        letters_counts.append(len(doc.text))

    return labels, word_counts, sents_counts, letters_counts


def visualize_counts(years, word_counts, sents_counts, letters_counts):

    df_dict = {'year': years, 'sents_count': sents_counts, 'words_count': word_counts, 'letters_count': letters_counts}
    counts_df = pd.DataFrame(df_dict)
    counts_df.sort_values('year', inplace=True)
    counts_df = counts_df.groupby(['year']).mean()  # Average counts per year

    axes = counts_df.plot(subplots=True, figsize=(13, 8), sharex=True, linewidth=2, legend=False)
    axes[0].set_ylabel('Mean number of letters')
    axes[1].set_ylabel('Mean number of sentences')
    axes[2].set_ylabel('Mean number of words')

    for i, ax in enumerate(axes):
        if i in [0, 1]:
            ax.xaxis.set_ticks_position('none')
        else:
            ax.xaxis.set_ticks((range(round_up(min(years), 10), round_up(max(years), 10), 10)))
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.show()


def get_pos_tags(corpus_dict, transform_label: Callable=None):
    """Calculates the number of words, sents and letters for each entry in corpus_dict.

    Arguments
    ---------
    corpus_dict: Dictionary of label-(raw)text key-value pairs.
    transform_label: Function to transform the labels from the corpus dict (e.g. extract year from label).

    Returns
    -------
    labels: Transformed or untransformed labels of the corpus_dict
    pos_tags: List of POS tags lists.
    """

    labels = []
    pos_tags = []

    for label, text in corpus_dict.items():
        if transform_label:
            label = transform_label(label)
        labels.append(label)

        pos_tags_ = [token.pos_ for token in spacy_doc(text)]
        pos_tags.append(pos_tags_)

    return labels, pos_tags


def aggregate_avg_pos_tags(years, pos_tags):

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
