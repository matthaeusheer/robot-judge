from collections import Counter


def normalize_counter(counter: Counter) -> Counter:
    """Normalizes a counter such that relative frequency is expressed rather than absolute counts."""
    total = sum(counter.values(), 0.0)
    for key in counter:
        counter[key] /= total
    return counter


def sort_coo_matrix(m):
    """Sorts a sparse COO format scipy matrix according to the value."""
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: x[2])[::-1]


def flatten_list_of_lists(list_of_lists):
    """Takes a list of lists and returns one list which holds all elements of all lists."""
    return [item for sublist in list_of_lists for item in sublist]


def get_most_n_most_common_counter_entries(counter_object, n_most_common):
    """The Counter most_common() method returns tuples of entries and counts. This method takes a counter and
    returns the n_most_common elements of this counter."""
    return [item[0] for item in counter_object.most_common()[:n_most_common]]
