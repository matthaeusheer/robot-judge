def round_down(num, divisor):
    """Returns the number num down to nearest divisor."""
    return num - (num % divisor)


def round_up(num, divisor):
    """Returns the number num up to nearest divisor."""
    return num + (num % divisor)
