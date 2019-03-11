import pandas as pd
import matplotlib.pyplot as plt

from robot_judge.nlp import count_sents
from robot_judge.nlp import count_tokens
from robot_judge.nlp import spacy_doc
import robot_judge.nlp.filter as filters


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


def visualize_counts(year, word_counts, sents_counts, letters_counts):

    df_dict = {'year': year, 'sents_count': sents_counts, 'words_count': word_counts, 'letters_count': letters_counts}
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
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.show()
