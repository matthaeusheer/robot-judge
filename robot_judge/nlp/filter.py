from robot_judge.nlp.language_models import spacy_nlp_en as nlp
from robot_judge.nlp.language_models import stop_words

"""
The following filter functions act on a complete spacy doc.
They return a cleaned version of the doc. Caution: Might be slow if applied 
multiple times to large corpora.
"""


def remove_punct_and_sym(doc):
    tokens = [tok.text for tok in doc if (tok.pos_ != "PUNCT" and tok.pos_ != "SYM")]
    return nlp.make_doc(' '.join(tokens))


def remove_ws_tokens(doc):
    tokens = [tok.text for tok in doc if not tok.is_space]
    return nlp.make_doc(' '.join(tokens))


def lemmatize(doc):
    tokens = [tok.lemma_.lower().strip() for tok in doc]
    return nlp.make_doc(' '.join(tokens))


def remove_stopwords(doc):
    tokens = [tok.text for tok in doc if str(tok) not in stop_words]
    return nlp.make_doc(' '.join(tokens))


def to_lower(doc):
    tokens = [token.text.lower() for token in doc]
    return nlp.make_doc(' '.join(tokens))


"""
The following filter functions act on a single token inside a spacy doc.
They return boolean True / False if the condition is met.
"""


def token_is_punct_space(token):
    return token.is_punct or token.is_space


def token_is_stopword(token):
    return token.text in stop_words
