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
