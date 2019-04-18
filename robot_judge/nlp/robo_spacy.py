from typing import Dict
from robot_judge import io as robo_io
from robot_judge.exploration import corpus_analysis
from robot_judge.nlp.language_models import spacy_nlp_en as nlp


def spacy_doc(text):
    """Runs the spacy language model pipeline on a text. Uses english language model."""
    return nlp(text)


def count_sents(doc):
    """Counts the sentences in a spacy document."""
    return sum(1 for _ in doc.sents)


def count_tokens(tokens_list):
    """Counts number of tokens in a list of tokens."""
    return sum(1 for _ in tokens_list)


def custom_pipeline():
    """Returns a tuple of pipeline steps for the spaCy doc pipeline."""
    return nlp.tagger, nlp.parser, nlp.entity


def preprocess_or_load(do_process: bool, do_load: bool, data_dir_name=None, n_sampled_cases=None) -> Dict:
    """Speed up development by not having to process whole corpus all the time.

    If there is a processed corpus already, load this instead of processing everything, saves time.
    """
    if do_process and not do_load:
        assert data_dir_name and n_sampled_cases, 'If you want to process, specify the data dir name and n samples.'
        io = robo_io.DataLoader(data_dir=data_dir_name)
        corpus_dict = io.read_multiple_cases_files(n_samples=n_sampled_cases, random_seed=1)
        doc_dict = corpus_analysis.process_corpus_dict_to_doc_dict(corpus_dict, store=False, n_threads=4)
        return doc_dict
    elif not do_process and do_load:
        doc_dict = corpus_analysis.load_doc_dict()
        return doc_dict
    else:
        raise ValueError('Chose either PROCESS or LOAD to be true.')

