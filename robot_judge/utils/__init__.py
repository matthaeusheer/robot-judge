import textwrap


def indent(text, amount, ch=' '):
    return textwrap.indent(text, amount * ch)