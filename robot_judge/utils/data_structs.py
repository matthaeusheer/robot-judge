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


