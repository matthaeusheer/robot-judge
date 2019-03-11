from robot_judge.nlp.language_models import spacy_nlp as nlp
from robot_judge.nlp.language_models import stop_words


def remove_punct_and_sym(doc):
    tokens = [tok.text for tok in doc if (tok.pos_ != "PUNCT" and tok.pos_ != "SYM")]
    return nlp.make_doc(' '.join(tokens))


def remove_ws_tokens(doc):
    tokens = [tok.text for tok in doc if tok.pos_ != '_SP']
    return nlp.make_doc(' '.join(tokens))


def lemmatize(doc):
    tokens = [tok.lemma_.lower().strip() for tok in doc]
    return nlp.make_doc(' '.join(tokens))


def remove_stopwords(doc):
    tokens = [tok.text for tok in doc if str(tok) not in stop_words]
    return nlp.make_doc(' '.join(tokens))