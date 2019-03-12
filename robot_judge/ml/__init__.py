

def transform_to_text_label(y, zero_label, one_label):
    return [one_label if item == 1 else zero_label for item in y]
