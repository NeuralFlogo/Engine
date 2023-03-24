import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def tokenize(text, language='english', lemmatize=False):
    stop = set(stopwords.words(language) + list(string.punctuation))
    if lemmatize:
        return [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(text.lower()) if word not in stop]
    return [word for word in word_tokenize(text.lower()) if word not in stop]
