# -*- coding: iso-8859-1 -*-

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

def remove_stopwords(text: str, stopwords_file: str = "data/stopwords.txt"):
    """
    This function takes in a string of text and removes stopwords from it. Stopwords are common words that are often
    removed from text data because they do not carry much meaning or significance. This function uses a combination of
    stopwords from a file and the built-in Spanish stopwords list from the nltk package.

    ### Parameters
    #### text: str
        The input text as a string.
    #### stopwords_file: str
        The path to the stopwords file. Default is "data/stopwords.txt".
    ### Return
    #### str
        The input text with stopwords removed as a string.
    """
    # Load stopwords from file and NLTK
    with open(stopwords_file, encoding='iso-8859-1') as f:
        file_stopwords = [word.strip() for word in f.readlines()]
    nltk_stopwords = stopwords.words("spanish")
    # Combine stopwords from both sources
    all_stopwords = set(file_stopwords + nltk_stopwords)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in all_stopwords]
    return " ".join(filtered_words)