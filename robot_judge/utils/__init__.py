import textwrap
from datetime import datetime


def indent(text, amount, ch=' '):
    return textwrap.indent(text, amount * ch)


def get_datetime_tag():
    """Returns a datetime string in the format YYYY-MM-DD-HH-MM-SS-ssssss."""
    current_time = datetime.now()

    time_str = str(current_time)
    for literal in (' ', ':', '.'):
        time_str = time_str.replace(literal, '-')

    return time_str
