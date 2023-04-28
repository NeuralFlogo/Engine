import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class textPreprocessor:

    def __init__(self, language="English", lemmatize=False):
        self.language = language
        self.lemmatize = lemmatize

    def process(self, text):
        stop = set(stopwords.words(self.language) + list(string.punctuation))
        if self.lemmatize and self.language == 'english':
            return [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(text.lower()) if word not in stop]
        return [word for word in word_tokenize(text.lower()) if word not in stop]
