from robot_judge.nlp.language_models import spacy_nlp as nlp


def spacy_doc(text):
    return nlp(text)


def count_sents(doc):
    return sum(1 for _ in doc.sents)


def count_tokens(doc):
    return sum(1 for _ in doc)
