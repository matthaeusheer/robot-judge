from wordcloud import WordCloud


def list_of_words_to_point_cloud(words, max_words):
    """Takes in a list of words and produces a point cloud object."""
    long_string = ' '.join(words)
    word_cloud = WordCloud(background_color="white", max_words=max_words, contour_width=3, contour_color='steelblue')
    return word_cloud.generate(long_string)
