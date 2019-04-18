from robot_judge.nlp import ngrams


def setup_feature_matrix(sents_dict, filtered_words, io_object, target):
    """Based on a sents_dict and filtered words list put up a feature matrix and target vector.

    Arguments
    ---------
        sents_dict: dict with labels as keys and lists of sentences (list of words) as values
        filtered_words: list of filtered words (e.g. most common tri and bi gram phrases)
        io_object: DataLoader instance to parse the metadata file
        target: which column of the metadata file to be used as prediction target
    """
    feat_df = ngrams.create_df_from_label_sent_dict(sents_dict)
    target_labels = ngrams.get_labels_without_year(feat_df)
    target_values = io_object.get_target_values(target_labels, target)
    feat_df.insert(0, f'__{target}__', target_values)
    feat_df.fillna(0.0, inplace=True)

    feat_df = feat_df[filtered_words + [f'__{target}__']]
    y = feat_df[f'__{target}__']
    X = feat_df.loc[:, feat_df.columns != f'__{target}__']
    return X, y
