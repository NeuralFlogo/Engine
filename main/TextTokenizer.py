import string

from nltk import word_tokenize
from nltk.corpus import stopwords


def tokenize(text, language='english'):
    stop = set(stopwords.words(language) + list(string.punctuation))
    return [word for word in word_tokenize(text.lower()) if word not in stop]
